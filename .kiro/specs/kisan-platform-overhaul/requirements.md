# Requirements Document

## Introduction

This document defines the requirements for a major overhaul of Project-Kisan, an existing agentic farming assistant built on a LangGraph workflow with a FastAPI backend, a relational user database, and a Qdrant vector store. The overhaul addresses correctness defects (broken short-term memory, dead/placeholder graph nodes, non-streaming output), introduces new capabilities (per-user isolated long-term memory, proper farmer-location modeling, phone-number authentication, multi-chat session management, a simple web frontend), migrates all persistence to managed cloud services (Neon Postgres for relational and LangGraph memory, Qdrant Cloud for vectors), and restructures the repository so the backend and database layers live under `src`.

The intent is to preserve existing working behavior (routing, agent responses, tool usage) while fixing identified defects and adding the listed features. Each requirement focuses on observable system behavior (the *what*), leaving implementation choices (the *how*) to the design phase.

## Glossary

- **Platform**: The complete Project-Kisan system, including the AI agent, backend API, databases, and frontend.
- **Agent_Graph**: The LangGraph state-machine workflow that routes a user query through nodes and produces a response.
- **Checkpointer**: The LangGraph persistence component that stores per-thread conversation state (short-term memory).
- **Long_Term_Store**: The LangGraph store that persists durable, per-user memories across threads and sessions.
- **Short_Term_Memory**: Conversation state scoped to a single thread, restored when the same thread is resumed.
- **Thread**: A single chat conversation identified by a unique thread identifier and owned by exactly one user.
- **Chat_Session**: A user's interaction with a single Thread, including its message history and metadata.
- **Backend_API**: The FastAPI service exposing authentication, chat, and user-management endpoints.
- **Auth_Service**: The Backend_API component responsible for registration, login, and token issuance/verification.
- **Vector_Store**: The Qdrant Cloud vector database storing embedded memories.
- **Relational_DB**: The Neon Postgres relational database storing user records and chat metadata.
- **Farmer_Directory**: The data store and query interface used to find nearby farmers by location.
- **Node**: A unit of work in the Agent_Graph (for example, WeatherNode, MandiNode, CarbonFootprintNode).
- **Placeholder_Node**: A Node that currently returns hardcoded or empty output without performing its intended function (CarbonFootprintNode, TextNode).
- **Streaming_Response**: The final assistant answer delivered to the client incrementally, token by token, over Server-Sent Events.
- **SSE**: Server-Sent Events delivered via the existing `sse_starlette` `EventSourceResponse` mechanism.
- **Frontend**: The web user interface providing login, signup, and chat screens.
- **Existing_API_Key**: An API key already present in the project configuration prior to this overhaul.
- **New_API_Key**: An API key not present in the project configuration prior to this overhaul.

## Requirements

### Requirement 1: Short-Term Memory Persistence via Checkpointer

**User Story:** As a farmer continuing a conversation, I want the assistant to remember what we discussed earlier in the same chat, so that I do not have to repeat context.

#### Acceptance Criteria

1. WHEN the Backend_API starts, THE Platform SHALL initialize the Checkpointer with a connection to the Neon Postgres database within 30 seconds.
2. WHEN a query is processed for a Thread, THE Checkpointer SHALL persist the resulting conversation state under that Thread identifier before the response is returned to the client.
3. WHEN a query is submitted to an existing Thread, THE Agent_Graph SHALL load the previously persisted conversation state for that Thread before generating a response.
4. WHEN two queries are submitted in sequence to the same Thread, THE Agent_Graph SHALL include the first query and its response in the conversation history available to the second query.
5. WHERE two queries reference different Thread identifiers, THE Checkpointer SHALL keep each Thread's conversation state isolated such that one Thread's history contains no messages from the other.
6. IF the Checkpointer connection cannot be established within 30 seconds at startup, THEN THE Backend_API SHALL halt initialization of the Checkpointer and log a descriptive error identifying the failed persistence component.
7. IF persisting a Thread's conversation state fails, THEN THE Backend_API SHALL return an error indication to the client and SHALL leave the Thread's previously persisted state unchanged.
8. IF loading a Thread's persisted conversation state fails, THEN THE Backend_API SHALL return an error indication to the client and SHALL NOT generate a response from incomplete state.

### Requirement 2: Resolution of Placeholder and Dead Graph Nodes

**User Story:** As a farmer using the assistant, I want every feature the agent advertises to either work or clearly tell me it is not yet available, so that I am not misled by fake responses.

#### Acceptance Criteria

1. WHERE a Placeholder_Node can perform its intended function using only Existing_API_Keys, THE Platform SHALL implement that Node to produce a response that varies when its input varies and that contains no fixed hardcoded result string, with no exceptions for any node that can use existing keys.
2. WHERE a Node requires a New_API_Key to perform its intended function, THE Node SHALL return a response containing the literal text "Coming soon" without attempting the operation or returning fabricated values.
3. WHEN the CarbonFootprintNode is invoked and its required data sources are available through Existing_API_Keys, THE CarbonFootprintNode SHALL return a response that varies with the conversation input and contains no fixed hardcoded result string.
4. WHEN the CarbonFootprintNode is invoked and its required data sources are unavailable without a New_API_Key, THE CarbonFootprintNode SHALL return a response containing the literal text "Coming soon" without returning fabricated values.
5. WHEN the TextNode is reached as the output Node for a text response, THE TextNode SHALL output the final assistant message exactly, with no characters added, removed, or replaced.
6. IF a Node is invoked without the input parameters required to perform its function, THEN THE Node SHALL return an error indication identifying the missing input without fabricating placeholder values.
7. IF a Node's required data source backed by an Existing_API_Key fails at invocation, THEN THE Node SHALL return a response indicating the data could not be retrieved without fabricating values.
8. THE Platform SHALL maintain a record listing, for each Node, whether the Node is implemented or marked "Coming soon".

### Requirement 3: Token-by-Token Streaming of Final Output

**User Story:** As a farmer waiting for an answer, I want the response to appear progressively as it is generated, so that the assistant feels responsive.

#### Acceptance Criteria

1. WHEN a client submits a chat request that sets the streaming option to enabled, THE Backend_API SHALL deliver the final assistant answer as a Streaming_Response over SSE.
2. WHEN the final assistant answer is generated, THE Backend_API SHALL emit the answer in incremental token chunks of between 1 and 512 characters each rather than as a single complete message.
3. WHILE the Agent_Graph is producing the final answer, THE Backend_API SHALL send each generated token chunk to the client before the full answer is complete.
4. WHEN a Streaming_Response begins, THE Backend_API SHALL deliver the first token chunk to the client within 5 seconds of the streaming chat request being accepted.
5. IF no token chunk is emitted for a continuous period of 30 seconds while a Streaming_Response is in progress, THEN THE Backend_API SHALL terminate the stream and emit an error event indicating a streaming timeout.
6. WHEN the final assistant answer is fully delivered, THE Backend_API SHALL emit a completion event indicating the end of the Streaming_Response, after which no further token chunks are sent for that response.
7. IF an error occurs during streaming, THEN THE Backend_API SHALL emit an error event over the same SSE stream indicating the failure cause and SHALL terminate the stream without emitting a completion event.
8. WHERE a client submits a chat request that sets the streaming option to disabled, THE Backend_API SHALL return the complete final answer in a single response.

### Requirement 4: Per-User Isolated Long-Term Memory

**User Story:** As a farmer, I want the assistant to remember durable facts about me across my different chats, while never mixing my information with another farmer's, so that advice stays personal and private.

#### Acceptance Criteria

1. THE Long_Term_Store SHALL persist durable memories (facts intended to outlive a single Thread, such as the user's crops, location, and stated preferences) in the Neon Postgres database, retaining each memory until it is explicitly deleted.
2. WHEN a memory is written for a user, THE Long_Term_Store SHALL associate that memory with a namespace derived from that user's unique user identifier.
3. WHEN long-term memories are retrieved for a user, THE Long_Term_Store SHALL return only memories whose namespace matches that user's unique user identifier.
4. WHERE two users have stored long-term memories, THE Long_Term_Store SHALL prevent one user's retrieval from returning any memory associated with the other user's namespace.
5. WHEN a long-term memory search is performed for a query, THE Long_Term_Store SHALL return matching memories from the requesting user's namespace ordered from highest to lowest relevance to the query, limited to a configurable maximum result count (default 10).
6. WHEN a long-term memory search for the requesting user's namespace finds no memories matching the query, THE Long_Term_Store SHALL return an empty result without error.
7. WHEN a conversation is determined to contain durable user information (as defined in criterion 1), THE Platform SHALL write that information to the Long_Term_Store under the corresponding user's namespace.
8. IF a long-term memory write fails, THEN THE Long_Term_Store SHALL return an error indication identifying the failed write and SHALL leave previously stored memories for that user unchanged.

### Requirement 5: Proper Farmer-Location Data Modeling and Nearby Search

**User Story:** As a farmer, I want to find other farmers near me who face similar issues, so that we can connect, without my profile data being duplicated across unrelated records.

#### Acceptance Criteria

1. THE Farmer_Directory SHALL store each user's location and contact data in a dedicated store that contains no per-conversation memory records.
2. WHEN a conversation summary is stored in the Vector_Store, THE Platform SHALL exclude all full user profile fields (phone number, name, user identifier, age, address) from that summary record's metadata.
3. WHEN a nearby-farmer search is requested for a user whose location record exists, THE Farmer_Directory SHALL return all farmers whose stored location lies within the requested search radius, where the radius is between 1 and 500 kilometers and defaults to 50 kilometers when unspecified.
4. WHEN a nearby-farmer search returns matching farmers, THE Farmer_Directory SHALL return at most 100 farmers ordered by ascending distance from the requesting user.
5. WHEN a user's profile is created or updated with location data, THE Farmer_Directory SHALL store exactly one authoritative location record per user, replacing any prior location record for that user.
6. WHERE a nearby-farmer search returns results, THE Farmer_Directory SHALL exclude the requesting user from those results.
7. WHEN the Farmer_Directory returns a nearby farmer, THE response SHALL include that farmer's phone number as the contact information required to reach the farmer.
8. IF a nearby-farmer search is requested for a user who has no stored location record, THEN THE Farmer_Directory SHALL reject the request with an error indicating that the requesting user's location is not set, and SHALL return no farmer records.
9. IF a nearby-farmer search finds no farmers within the requested radius, THEN THE Farmer_Directory SHALL return an empty result set rather than an error.

### Requirement 6: Phone-Number-Based Authentication

**User Story:** As a farmer who will be contacted by phone, I want to register and sign in using my phone number, so that my account is tied to the number used to reach me.

#### Acceptance Criteria

1. WHEN a user submits a registration request with a phone number in valid format and a password of 8 to 128 characters, THE Auth_Service SHALL create exactly one user account keyed to that phone number and return a confirmation of successful registration.
2. WHEN a user submits a registration phone number that already belongs to an existing account, THE Auth_Service SHALL reject the registration without creating a new account and return an error indicating the phone number is already registered.
3. WHEN a user submits a registered phone number together with a password that matches the stored credential, THE Auth_Service SHALL authenticate the user and issue an access token that expires 15 minutes after issuance and a refresh token that expires 7 days after issuance.
4. IF a user submits a phone number that is not registered or a password that does not match the stored credential, THEN THE Auth_Service SHALL reject the login without issuing any token and return an authentication error that does not disclose which of the two fields was incorrect.
5. WHEN a phone number is submitted for registration or login, THE Auth_Service SHALL validate the phone number against E.164 format (a leading "+" followed by 8 to 15 digits) before processing the request.
6. IF a submitted phone number does not conform to E.164 format, THEN THE Auth_Service SHALL reject the request without creating an account or issuing any token and return an error indicating the phone number format is invalid.
7. WHEN an access token has expired and a valid, unexpired refresh token is submitted, THE Auth_Service SHALL issue a new access token that expires 15 minutes after issuance.
8. IF an expired, malformed, or unrecognized refresh token is submitted, THEN THE Auth_Service SHALL reject the request without issuing a new access token and return an authentication error.

### Requirement 7: Per-User Multi-Chat Session Management

**User Story:** As a farmer, I want to keep multiple separate chats and switch between them, so that I can organize different topics.

#### Acceptance Criteria

1. WHEN an authenticated user creates a new chat, THE Backend_API SHALL generate a Thread identifier that is unique across the Platform and associate the Thread with that user's identity.
2. WHEN an authenticated user requests the list of chats, THE Backend_API SHALL return all Threads owned by that user ordered from most recently active to least recently active, and SHALL return an empty list when the user owns no Threads.
3. WHEN an authenticated user requests a chat the user owns, THE Backend_API SHALL return the ordered message history for that Thread from oldest to newest message.
4. WHEN an authenticated user deletes a chat the user owns, THE Backend_API SHALL remove that Thread and its persisted conversation state so that subsequent requests for that Thread return a not-found result.
5. IF a user requests or deletes a Thread that does not exist or that the user does not own, THEN THE Backend_API SHALL reject the request with an authorization error and SHALL leave all stored Threads and conversation state unchanged.
6. WHEN the first user message is sent in a newly created chat, THE Backend_API SHALL assign the chat a name composed of the leading characters of that first message truncated to a maximum of 50 characters.
7. IF the first user message in a newly created chat contains no non-whitespace characters, THEN THE Backend_API SHALL assign the chat a default name.
8. WHEN a chat is listed, THE Backend_API SHALL include the chat name and the total count of messages in the chat as a non-negative integer.

### Requirement 8: Web Frontend for Authentication and Chat

**User Story:** As a farmer, I want a simple web interface to sign up, log in, and chat, so that I can use the assistant without technical setup.

#### Acceptance Criteria

1. THE Frontend SHALL provide a signup screen that collects a phone number and a password and submits them as registration data to the Auth_Service.
2. THE Frontend SHALL provide a login screen that submits a phone number and password to the Auth_Service.
3. WHEN a user is authenticated, THE Frontend SHALL display a chat screen with a sidebar listing the user's chats.
4. WHEN a user selects a chat from the sidebar, THE Frontend SHALL load and display that chat's message history.
5. WHEN a user sends a message, THE Frontend SHALL render each token chunk of the assistant's response as it is received over the Streaming_Response, before the full response is complete.
6. WHEN a user requests deletion of a chat and the Backend_API confirms the deletion, THE Frontend SHALL remove the chat from the sidebar.
7. WHEN a user starts a new chat, THE Frontend SHALL create a new Thread through the Backend_API and make it the active chat.
8. IF an authentication request fails, THEN THE Frontend SHALL display an error message that indicates the specific reason for the failure (such as invalid credentials or a phone number already registered), including when the failure is that a phone number is already registered.
9. IF the Backend_API does not confirm a requested chat deletion, THEN THE Frontend SHALL retain the chat in the sidebar and display an error message indicating that the deletion failed.
10. IF an explicit error event is received over the Streaming_Response, THEN THE Frontend SHALL stop rendering token chunks and display an error message indicating that the response failed; connection loss or timeout without an explicit error event SHALL NOT trigger this stop-rendering behavior.
11. IF a request to load a selected chat's message history fails, THEN THE Frontend SHALL retain the currently displayed chat and display an error message indicating that the history could not be loaded.

### Requirement 9: Cloud Database Migration

**User Story:** As the platform operator, I want all databases hosted on managed cloud services, so that data persists reliably and is accessible without local infrastructure.

#### Acceptance Criteria

1. THE Relational_DB SHALL use the Neon Postgres connection string supplied through the `NEON_API` environment variable.
2. THE Checkpointer SHALL persist Short_Term_Memory in the Neon Postgres database.
3. THE Long_Term_Store SHALL persist long-term memories in the Neon Postgres database.
4. THE Vector_Store SHALL connect to Qdrant Cloud using the URL from the `QDRANT_URL` environment variable and the API key from the `QDRANT_API` environment variable.
5. WHEN the Platform initializes a database connection, THE Platform SHALL read the connection settings exclusively from the `NEON_API`, `QDRANT_URL`, and `QDRANT_API` environment variables and SHALL NOT use any hardcoded local file path or local host address.
6. IF a required database environment variable (`NEON_API`, `QDRANT_URL`, or `QDRANT_API`) is unset or empty at startup, THEN THE Platform SHALL stop initialization of the component that depends on that variable, SHALL log an error message naming the missing variable, and SHALL NOT fall back to a local database.
7. THE Platform SHALL establish all Vector_Store connections through an authenticated Qdrant Cloud client rather than an unauthenticated local client.
8. IF a database connection cannot be established within 30 seconds of an initialization attempt while the required environment variables are set, THEN THE Platform SHALL stop initialization of the affected component and log a descriptive error identifying the affected database.
9. WHEN all required database environment variables are set and their connections are established successfully at startup, THE Platform SHALL complete initialization of the Relational_DB, Checkpointer, Long_Term_Store, and Vector_Store before the Backend_API accepts client requests.

### Requirement 10: Repository Restructure Under `src`

**User Story:** As a developer maintaining the project, I want the backend and database layers organized under `src`, so that the codebase has a consistent structure and imports resolve correctly.

#### Acceptance Criteria

1. THE Platform SHALL locate all backend application code under the `src` directory such that no backend module remains outside `src`.
2. THE Platform SHALL locate all database layer code under the `src` directory such that no database module remains outside `src`.
3. WHEN the backend and database code are relocated under `src`, THE Platform SHALL update module imports such that no import statement resolves to a pre-restructure path; IF the automated import update fails partway through, THEN THE Platform SHALL roll back the entire restructure, restoring all files to their pre-restructure locations and leaving no broken imports.
4. WHEN the Backend_API is started after the restructure, THE Backend_API SHALL import all relocated modules to completion without raising any import error, and THE Backend_API SHALL be verified as successfully started before import completion is confirmed.
5. IF importing a relocated module fails at startup, THEN THE Backend_API SHALL halt initialization and log a descriptive error naming the module that failed to import.
6. THE Platform SHALL preserve, for each relocated module, the same public interfaces and the same observable outputs for identical inputs after the restructure.

### Requirement 11: Preserve Existing Agent Capabilities

**User Story:** As a farmer who already relies on the assistant, I want the existing features to keep working after the overhaul, so that the upgrade does not break what I use today.

#### Acceptance Criteria

1. WHEN a query is submitted, THE Agent_Graph SHALL route the query to exactly one Node corresponding to one of the supported workflows (General, Disease, Weather, Mandi, Government Scheme) using the existing routing logic.
2. WHEN a routed Node requires external data, THE Agent_Graph SHALL invoke the corresponding tools for that Node.
3. WHEN the invoked tools return results, THE Agent_Graph SHALL incorporate those tool results into the response returned to the client.
4. WHEN a query is routed to the General, Disease, Weather, Mandi, or Government Scheme workflow, THE Agent_Graph SHALL produce a non-empty response for that workflow.
5. WHERE a workflow Node is unchanged by this overhaul, THE Platform SHALL produce, for an identical query and conversation state, a response equivalent to that Node's pre-overhaul response behavior, and this strict equivalence SHALL hold regardless of changes to external dependencies or system state.
6. IF the existing routing logic cannot match a submitted query to any supported workflow, THEN THE Agent_Graph SHALL route the query to the General workflow as the default Node.
7. IF a tool invocation fails or returns no usable data, THEN THE Agent_Graph SHALL produce a response indicating that the external data could not be retrieved and SHALL continue the conversation without terminating the Thread; an empty response is acceptable when data retrieval fails even if the tool invocation itself did not fail.
