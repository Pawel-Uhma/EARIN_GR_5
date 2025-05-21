Prolog Working Days Calculator (2024)
This Prolog program calculates the weekday and date after N working days from a given date in 2024, skipping weekends.

Usage

1. Load in SWI-Prolog

   Open a terminal and run:

2. Run a query

   ?- n_work_days('2205', 6, R).
   R = 'Thursday, 3005'.

   n_work_days(DateAtom, N, Result).
   DateAtom: an atom like 'DDMM' (e.g., '2205')

   N: number of working days to add

   Result: output atom like 'Thursday, 3005'

Notes:
Works only for 2024
Skips Saturdays and Sundays
Does not account for public holidays
