import re

import pandas as pd
import pytz

class StaticTools:
    def __init__(self):
        self.timezone_mapping = {
            'pt': 'America/Los_Angeles',
            'et': 'America/New_York',
            'mt': 'America/Denver',
            'ct': 'America/Chicago',
            'ak': 'America/Anchorage',
            'hat': 'America/Adak'
        }

    @staticmethod
    def is_number(value):
        try:
            # Try converting to an integer
            int(value)
            return True  # If this succeeds, it's an integer
        except ValueError:
            try:
                # Try converting to a float if int fails
                float(value)
                return True  # If this succeeds, it's a float (or a number with decimal)
            except ValueError:
                return False  # If both fail, it's not a number
    @staticmethod
    def string_to_datetime(text, timezone="utc"):
        # print(f"start fixing date string {text}")
        text = re.sub(r"[.,]", "", text)  # Remove periods and commas
        # Timezone objects
        pt_timezone = pytz.timezone('America/Los_Angeles')
        et_timezone = pytz.timezone('America/New_York')
        mt_timezone = pytz.timezone('America/Denver')
        ct_timezone = pytz.timezone('America/Chicago')
        ak_timezone = pytz.timezone('America/Anchorage')
        hat_timezone = pytz.timezone('America/Adak')
        if 'pdt' in text or 'pt' in text:
            text = text.replace("pdt", "").replace("pt", "")
            parsed_date = pd.to_datetime(text, errors='coerce').tz_localize(pt_timezone, ambiguous='NaT')
        elif 'et' in text or 'edt' in text or 'est' in text:
            text = text.replace("et", "").replace("edt", "").replace("est", "")
            parsed_date = pd.to_datetime(text, errors='coerce').tz_localize(et_timezone, ambiguous='NaT')
        elif 'mt' in text or 'mst' in text:
            text = text.replace("mt", "").replace("mst", "")
            parsed_date = pd.to_datetime(text, errors='coerce').tz_localize(mt_timezone, ambiguous='NaT')
        elif 'ct' in text or 'cst' in text or 'cdt' in text:
            text = text.replace("ct", "").replace("cst", "").replace("cdt", "")
            parsed_date = pd.to_datetime(text, errors='coerce').tz_localize(ct_timezone, ambiguous='NaT')
        elif 'ak' in text or 'akdt' in text or 'akst' in text:
            text = text.replace("ak", "").replace("akdt", "").replace("akst", "")
            parsed_date = pd.to_datetime(text, errors='coerce').tz_localize(ak_timezone, ambiguous='NaT')
        elif 'hat' in text or 'hst' in text:
            text = text.replace("hat", "").replace("hst", "")
            parsed_date = pd.to_datetime(text, errors='coerce').tz_localize(hat_timezone, ambiguous='NaT')
        else:
            parsed_date = pd.to_datetime(text, errors='coerce')
            parsed_date = parsed_date.tz_localize("utc")

        # Convert everything to UTC
        parsed_date = parsed_date.tz_convert(timezone)
        if not pd.isna(parsed_date):
            print(f" got date from pd.to_datetime {parsed_date}")
            return parsed_date

        # Fix the format if it's missing a space between the date and time part
        if re.match(r'\d{2}/\d{2}/\d{4}\d{2}:\d{2} [APap][Mm]', text):
            text = re.sub(r'(\d{2}/\d{2}/\d{4})(\d{2}:\d{2})', r'\1 \2', text)
            print(f"Adjusted date format: {text}")

        # List of possible date formats for `pd.to_datetime`
        formats = [
            '%m/%d/%Y %I:%M %p',  # 'MM/DD/YYYY HH:MM AM/PM'
            '%Y-%m-%d %H:%M:%S',  # 'YYYY-MM-DD HH:MM:SS'
            '%m/%d/%Y',  # 'MM/DD/YYYY'
            '%d/%m/%Y',  # 'DD/MM/YYYY'
            '%B %d, %Y',  # 'Month DD, YYYY'
            '%d %B %Y',  # 'DD Month YYYY'
            '%Y-%m-%d',  # 'YYYY-MM-DD'
            '%I:%M %p',  # 'HH:MM AM/PM' (time only)
            '%H:%M',  # 'HH:MM' (24-hour time)
            '%Y/%m/%d %H:%M',  # 'YYYY/MM/DD HH:MM'
            '%Y/%m/%d',  # 'YYYY/MM/DD'
            '%a %b %d %Y %I:%M %p'  # 'Thu Aug 1 2024 11:19 AM'
        ]

        # Try all the date formats and attempt localization based on text
        for fmt in formats:
            try:
                parsed_date = pd.to_datetime(text, format=fmt, errors='raise')
                # Convert everything to UTC
                parsed_date = parsed_date.tz_convert('UTC')
                if not pd.isna(parsed_date):
                    print(f"completed fixing date string {text}")
                    return parsed_date
            except ValueError:
                continue  # Try the next format if ValueError occurs
                # If none of the formats work, return NaT (not a timestamp)
        return parsed_date