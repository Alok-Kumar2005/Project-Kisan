import sys
import os
from datetime import datetime
from typing import Dict, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from src.ai_component.core.schedules import (
    FRIDAY_SCHEDULE,
    MONDAY_SCHEDULE,
    SATURDAY_SCHEDULE,
    SUNDAY_SCHEDULE,
    THURSDAY_SCHEDULE,
    TUESDAY_SCHEDULE,
    WEDNESDAY_SCHEDULE,
)
from src.ai_component.logger import logging
from src.ai_component.exception import CustomException


class ScheduleContextGenerator:
    """Class to generate context about Ramesh Kumar current activity based on schedules."""

    SCHEDULES = {
        0: MONDAY_SCHEDULE,  # Monday
        1: TUESDAY_SCHEDULE,  # Tuesday
        2: WEDNESDAY_SCHEDULE,  # Wednesday
        3: THURSDAY_SCHEDULE,  # Thursday
        4: FRIDAY_SCHEDULE,  # Friday
        5: SATURDAY_SCHEDULE,  # Saturday
        6: SUNDAY_SCHEDULE,  # Sunday
    }

    @staticmethod
    def _parse_time_range(time_range: str) -> tuple:
        """Parse a time range string (e.g., '06:00-07:00') into start and end times."""
        try:
            logging.debug(f"Parsing time range: {time_range}")
            start_str, end_str = time_range.split("-")
            start_time = datetime.strptime(start_str, "%H:%M").time()
            end_time = datetime.strptime(end_str, "%H:%M").time()
            logging.debug(f"Parsed start time: {start_time}, end time: {end_time}")
            return start_time, end_time
        except CustomException as e:
            logging.error(f"Error in Engineering Node : {str(e)}")
            raise CustomException(e, sys) from e

    @classmethod
    def get_current_activity(cls) -> Optional[str]:
        """Get Ramesh Kumar current activity based on the current time and day of the week.

        Returns:
            str: Description of current activity, or None if no matching time slot is found
        """
        try:
            logging.info("Getting current activity for Ramesh Kumar")
            current_datetime = datetime.now()
            current_time = current_datetime.time()
            current_day = current_datetime.weekday()

            # Get schedule for current day
            schedule = cls.SCHEDULES.get(current_day, {})

            # Find matching time slot
            for time_range, activity in schedule.items():
                start_time, end_time = cls._parse_time_range(time_range)

                # Handle overnight activities (e.g., 23:00-06:00)
                if start_time > end_time:
                    if current_time >= start_time or current_time <= end_time:
                        return activity
                else:
                    if start_time <= current_time <= end_time:
                        return activity

            return None
        except CustomException as e:
            logging.error(f"Error in Engineering Node : {str(e)}")
            raise CustomException(e, sys) from e

    @classmethod
    def get_schedule_for_day(cls, day: int) -> Dict[str, str]:
        """Get the complete schedule for a specific day.

        Args:
            day: Day of week as integer (0 = Monday, 6 = Sunday)

        Returns:
            Dict[str, str]: Schedule for the specified day
        """
        try:
            logging.info(f"Getting schedule for day: {day}")
            return cls.SCHEDULES.get(day, {})
        except CustomException as e:
            logging.error(f"Error in Engineering Node : {str(e)}")
            raise CustomException(e, sys) from e
    

if __name__ == "__main__":
    # Example usage
    current_activity = ScheduleContextGenerator.get_current_activity()
    if current_activity:
        print(f"Ramesh Kumar current activity: {current_activity}")
    else:
        print("Ramesh Kumar is currently not scheduled for any activity.")
    
    # # Get schedule for a specific day (e.g., Monday)
    # monday_schedule = ScheduleContextGenerator.get_schedule_for_day(0)
    # print("Monday's schedule:", monday_schedule)