% month_days(Month, Days) says how many days are in each month of 2024
month_days(1, 31).
month_days(2, 29).
month_days(3, 31).
month_days(4, 30).
month_days(5, 31).
month_days(6, 30).
month_days(7, 31).
month_days(8, 31).
month_days(9, 30).
month_days(10, 31).
month_days(11, 30).
month_days(12, 31).

% sum_month_days(Month, Sum) calculates how many days have passed from 
% the beginning of the year up to the start of the given month.
% For example, for March (Month = 3), it adds days in January and February.

% Base case: if the month is January, no days have passed before it
sum_month_days(1, 0).

% Recursive case: for Month > 1
sum_month_days(M, Sum) :-
    M > 1,                 % make sure the month is greater than 1
    M1 is M - 1,           % look at the previous month
    sum_month_days(M1, S1),% recursively calculate days up to previous month
    month_days(M1, D1),    % get how many days are in the previous month
    Sum is S1 + D1.        % total = days before + days in previous month


% day_of_year(Month, Day, DayOfYear) finds which day of the year it is.
% For example, day_of_year(3, 1, X) means "What number is March 1st?" → X = 61
% It adds the number of days in the months before, plus the given day.

day_of_year(M, D, DoY) :-
    sum_month_days(M, S),   % get total days before this month
    DoY is S + D.           % add the day of the current month to get the total


% weekday_number(Day, Month, Num) tells which day of the week it is in 2024.
% The result Num is a number: 1 = Monday, 2 = Tuesday, ..., 7 = Sunday.
% It uses the fact that January 1, 2024 is a Monday (which is day 1).

weekday_number(D, M, Num) :-
    day_of_year(M, D, DoY),        % find the number of the day in the year (e.g. Jan 2 -> 2)
    Num is ((DoY - 1) mod 7) + 1.  % subtract 1 to align Jan 1 = 0, then mod 7 to cycle through week
                                   % finally add 1 to shift result back to 1–7 (1=Monday)


% weekday_name(Number, Name) gives the name for each number
weekday_name(1, 'Monday').
weekday_name(2, 'Tuesday').
weekday_name(3, 'Wednesday').
weekday_name(4, 'Thursday').
weekday_name(5, 'Friday').
weekday_name(6, 'Saturday').
weekday_name(7, 'Sunday').

% next_date(Day, Month, NextDay, NextMonth) finds the next calendar day
% It handles moving to the next day, and also changing the month if needed
% It assumes all dates are in the year 2024, so it doesn't go to the next year

% Case 1: if it's not the last day of the month, just add 1 to the day
next_date(D, M, D1, M) :-
    month_days(M, MD),  % get the number of days in this month
    D < MD,             % check if we are before the last day
    !,                  % cut: we found the correct rule, don't try others
    D1 is D + 1.        % next day is just current day + 1

% Case 2: if it's the last day of the month, move to the 1st of next month
next_date(_, M, 1, M1) :-
    M < 12,             % check if it's not December yet
    !,                  % cut: this rule handles all months except December
    M1 is M + 1.        % next month is current month + 1

% Case 3: if it's December 31st, wrap around to January 1st
next_date(_, 12, 1, 1).  % go from December to January


% advance_work_days(Day, Month, N, OutDay, OutMonth)
% This counts N working days after a given starting date (Day, Month)
% It skips Saturdays and Sundays (which are not working days)

% Base case: if N is 0, we've counted all working days. Return current day.
advance_work_days(D, M, 0, D, M) :- 
    !.  % stop recursion when no more workdays to count

% Case 1: next day is Saturday → skip it, do not count it as a working day
advance_work_days(D, M, N, OutD, OutM) :-
    N > 0,
    next_date(D, M, D1, M1),           % get the next calendar day
    weekday_number(D1, M1, 6),        % check if it's Saturday (day 6)
    !,
    advance_work_days(D1, M1, N, OutD, OutM).  % recurse without changing N

% Case 2: next day is Sunday → skip it, do not count it
advance_work_days(D, M, N, OutD, OutM) :-
    N > 0,
    next_date(D, M, D1, M1),           % get the next calendar day
    weekday_number(D1, M1, 7),        % check if it's Sunday (day 7)
    !,
    advance_work_days(D1, M1, N, OutD, OutM).  % recurse without changing N

% Case 3: next day is a weekday (Mon–Fri) → count it as one working day
advance_work_days(D, M, N, OutD, OutM) :-
    N > 0,
    next_date(D, M, D1, M1),           % get the next calendar day
    N1 is N - 1,                       % reduce number of days to count
    advance_work_days(D1, M1, N1, OutD, OutM).  % recurse with N - 1

% pad2(Number, Atom) turns a number like 5 into '05', and 12 stays '12'
% This helps build proper two-digit date strings like '3005'
pad2(N, S) :-
    number_chars(N, Chs),               % turn number into list of characters
    ( Chs = [C] ->                      % if it's a single digit
        Chs2 = ['0', C]                 % add a '0' in front
    ; Chs2 = Chs                        % if already two digits, leave it
    ),
    atom_chars(S, Chs2).                % turn character list back into an atom

% n_work_days(DateAtom, N, ResultAtom)
% Main function to calculate the result
% DateAtom is a string like '2205' (22nd May), N is number of workdays
% ResultAtom will be a string like 'Thursday, 3005'
n_work_days(DateAtom, N, ResultAtom) :-
    atom_chars(DateAtom, [A, B, C, D]),        % split the date into characters
    number_chars(D0, [A, B]),                  % convert first two to a number (day)
    number_chars(M0, [C, D]),                  % convert last two to a number (month)
    advance_work_days(D0, M0, N, Df, Mf),      % calculate new date after N workdays
    weekday_number(Df, Mf, Wn),                % get number of weekday
    weekday_name(Wn, Wname),                   % get name of the weekday (e.g. Thursday)
    pad2(Df, DDpad),                           % make sure day is two digits
    pad2(Mf, MMpad),                           % make sure month is two digits
    atomic_list_concat([Wname, ', ', DDpad, MMpad], ResultAtom).  % combine into result

