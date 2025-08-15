import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import os
import asyncio
import aiosqlite
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

class ChatHistoryManager:
    """Utility class for managing chat history in SQLite database"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Store database in the project's data directory
            project_root = os.path.join(os.path.dirname(__file__), '..', '..')
            data_dir = os.path.join(project_root, 'data')
            db_path = os.path.join(data_dir, 'chat_history.db')
        
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    async def get_connection(self) -> aiosqlite.Connection:
        """Get async database connection"""
        return await aiosqlite.connect(self.db_path, check_same_thread=False)
    
    async def cleanup_old_threads(self, days_old: int = 30) -> int:
        """Clean up threads older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        async with await self.get_connection() as conn:
            # First, let's check what tables exist
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = await cursor.fetchall()
            print(f"Available tables: {[table[0] for table in tables]}")
            
            # The actual table name might be different, let's check the schema
            if any('checkpoint' in table[0].lower() for table in tables):
                # Find the correct checkpoint table
                checkpoint_table = next((table[0] for table in tables if 'checkpoint' in table[0].lower()), None)
                
                if checkpoint_table:
                    try:
                        # Get table schema first
                        cursor = await conn.execute(f"PRAGMA table_info({checkpoint_table})")
                        columns = await cursor.fetchall()
                        column_names = [col[1] for col in columns]
                        
                        # Use different date column based on what's available
                        date_column = 'created_at' if 'created_at' in column_names else 'timestamp'
                        if date_column not in column_names:
                            # If no date column, we can't do date-based cleanup
                            print("No date column found for cleanup")
                            return 0
                        
                        cursor = await conn.execute(
                            f"DELETE FROM {checkpoint_table} WHERE {date_column} < ?",
                            (cutoff_date.isoformat(),)
                        )
                        deleted_count = cursor.rowcount
                        await conn.commit()
                        return deleted_count
                    except Exception as e:
                        print(f"Error during cleanup: {e}")
                        return 0
            
            return 0
    
    async def get_user_thread_count(self, user_id: str) -> int:
        """Get the number of threads for a specific user"""
        async with await self.get_connection() as conn:
            try:
                # First check what tables exist
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = await cursor.fetchall()
                
                checkpoint_table = next((table[0] for table in tables if 'checkpoint' in table[0].lower()), None)
                
                if checkpoint_table:
                    cursor = await conn.execute(
                        f"""
                        SELECT COUNT(DISTINCT thread_id) 
                        FROM {checkpoint_table} 
                        WHERE thread_id LIKE ?
                        """,
                        (f"user_{user_id}_%",)
                    )
                    result = await cursor.fetchone()
                    return result[0] if result else 0
                
                return 0
            except Exception as e:
                print(f"Error getting user thread count: {e}")
                return 0
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get general database statistics"""
        async with await self.get_connection() as conn:
            try:
                # Get table info
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                tables = await cursor.fetchall()
                table_names = [table[0] for table in tables]
                
                checkpoint_table = next((table for table in table_names if 'checkpoint' in table.lower()), None)
                
                stats = {
                    "tables": table_names,
                    "database_path": self.db_path,
                    "database_exists": os.path.exists(self.db_path)
                }
                
                if checkpoint_table:
                    # Total checkpoints
                    cursor = await conn.execute(f"SELECT COUNT(*) FROM {checkpoint_table}")
                    total_checkpoints = (await cursor.fetchone())[0]
                    stats["total_checkpoints"] = total_checkpoints
                    
                    # Unique threads
                    cursor = await conn.execute(f"SELECT COUNT(DISTINCT thread_id) FROM {checkpoint_table}")
                    unique_threads = (await cursor.fetchone())[0]
                    stats["unique_threads"] = unique_threads
                else:
                    stats["total_checkpoints"] = 0
                    stats["unique_threads"] = 0
                
                # Database size
                if os.path.exists(self.db_path):
                    db_size = os.path.getsize(self.db_path)
                    stats["database_size_bytes"] = db_size
                    stats["database_size_mb"] = round(db_size / (1024 * 1024), 2)
                else:
                    stats["database_size_bytes"] = 0
                    stats["database_size_mb"] = 0
                
                return stats
            
            except Exception as e:
                return {
                    "error": str(e),
                    "database_path": self.db_path,
                    "database_exists": os.path.exists(self.db_path)
                }
    
    async def export_thread_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """Export all messages from a thread in a structured format"""
        try:
            from src.ai_component.graph.graph import get_thread_messages
            
            messages = await get_thread_messages(thread_id)
            
            exported_messages = []
            for i, msg in enumerate(messages):
                if hasattr(msg, 'content') and msg.content:
                    # Skip system messages
                    if hasattr(msg, 'type') and getattr(msg, 'type', None) == 'system':
                        continue
                    
                    exported_messages.append({
                        "sequence": i + 1,
                        "role": 'user' if hasattr(msg, 'type') and getattr(msg, 'type', None) == 'human' else 'assistant',
                        "content": msg.content,
                        "timestamp": getattr(msg, 'additional_kwargs', {}).get('timestamp', datetime.now().isoformat()),
                        "metadata": getattr(msg, 'additional_kwargs', {})
                    })
            
            return exported_messages
        
        except Exception as e:
            print(f"Error exporting thread messages: {e}")
            return []
    
    async def backup_user_data(self, user_id: str, backup_path: str) -> bool:
        """Backup all data for a specific user"""
        try:
            from src.ai_component.graph.graph import retrieve_all_threads_for_user
            
            # Get all threads for the user
            thread_ids = await retrieve_all_threads_for_user(user_id)
            
            user_backup = {
                "user_id": user_id,
                "backup_timestamp": datetime.now().isoformat(),
                "threads": {}
            }
            
            # Export each thread
            for thread_id in thread_ids:
                messages = await self.export_thread_messages(thread_id)
                user_backup["threads"][thread_id] = messages
            
            # Save backup to file
            import json
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(user_backup, f, indent=2, ensure_ascii=False)
            
            return True
        
        except Exception as e:
            print(f"Error backing up user data: {e}")
            return False
    
    async def get_thread_analytics(self, thread_id: str) -> Dict[str, Any]:
        """Get analytics for a specific thread"""
        try:
            messages = await self.export_thread_messages(thread_id)
            
            if not messages:
                return {"error": "No messages found"}
            
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
            
            # Calculate average response time (if timestamps are available)
            response_times = []
            for i in range(len(user_messages)):
                if i < len(assistant_messages):
                    try:
                        user_time = datetime.fromisoformat(user_messages[i]["timestamp"])
                        assistant_time = datetime.fromisoformat(assistant_messages[i]["timestamp"])
                        response_time = (assistant_time - user_time).total_seconds()
                        if response_time > 0:
                            response_times.append(response_time)
                    except:
                        continue
            
            # Calculate word counts
            user_word_count = sum(len(msg["content"].split()) for msg in user_messages)
            assistant_word_count = sum(len(msg["content"].split()) for msg in assistant_messages)
            
            return {
                "thread_id": thread_id,
                "total_messages": len(messages),
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages),
                "user_word_count": user_word_count,
                "assistant_word_count": assistant_word_count,
                "average_response_time_seconds": sum(response_times) / len(response_times) if response_times else 0,
                "conversation_duration_minutes": self._calculate_conversation_duration(messages),
                "first_message_time": messages[0]["timestamp"] if messages else None,
                "last_message_time": messages[-1]["timestamp"] if messages else None
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_conversation_duration(self, messages: List[Dict]) -> float:
        """Calculate total conversation duration in minutes"""
        if len(messages) < 2:
            return 0
        
        try:
            first_time = datetime.fromisoformat(messages[0]["timestamp"])
            last_time = datetime.fromisoformat(messages[-1]["timestamp"])
            duration = (last_time - first_time).total_seconds() / 60
            return round(duration, 2)
        except:
            return 0


# Global instance
chat_history_manager = ChatHistoryManager()