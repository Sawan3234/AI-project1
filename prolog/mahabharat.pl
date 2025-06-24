% Define basic facts about characters and their relationships

% Character facts
character(krishna).
character(arjun).
character(duryodhan).
character(bhim).
character(yudhishthir).
character(draupadi).
character(karn).

% Kauravas (100 sons of Dhritarashtra)
character(duhsasan).
character(duhsasan).
% Add all 100 Kauravas similarly, for brevity, only a few shown here
% and are generally used for the context.

% Relationships
% format: relation(Character1, Character2)

% Family Relations
father(yudhishthir, dharmaraj).
father(bhim, dharmaraj).
father(arjun, dharmaraj).

fatherinlaw(draupadi, dharmaraj).

motherinlaw(draupadi, kunti).

% Friendships
friend(arjun, krishna).
friend(krishn, arjun).

friend(bhim, krishna).

% Rivalries
rival(duryodhan, arjun).
rival(duryodhan, bhim).
rival(duryodhan, yudhishthir).
rival(karn,arjun).

% Mentor Relations
mentor(krishna, arjun).
mentor(bheeshma,arjun).
mentor(bheeshma, yudhishthir).

% Killing Relations
% format: killed(Killer, Victim)

killed(arjun, karn).
killed(bhim, duryodhan).
 killed(arjun,bheeshma).

% Example Queries

% Check if a given character is a friend of another character
is_friend(X, Y) :-
    friend(X, Y);
    friend(Y, X).

% Check if a character is a mentor of another character
is_mentor(X, Y) :-
    mentor(X, Y).

% Check who killed whom
who_killed(X, Y) :-
    killed(X, Y).

% Example queries and their usage
% ?- is_friend(arjuna, krishna).
% true.

% ?- is_mentor(krishna, arjuna).
% true.

% ?- who_killed(arjuna, Who).
% Who = karna.

% ?- who_killed(bhima, duryodhana).
% true.
