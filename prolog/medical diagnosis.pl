%defining facts
% Facts about symptoms
symptom(john, fever).
symptom(john, cough).
symptom(sue, headache).

% Rules to determine if a person has flu or cold
has_flu(X) :-
    symptom(X, fever),
    symptom(X, cough).

has_cold(X) :-
    symptom(X, headache),
    symptom(X, cough).

% General rule to check diagnosis based on symptoms
diagnose(X, flu) :-
    has_flu(X).

diagnose(X, cold) :-
    has_cold(X).
