%prolog program to implement friend of friend
% Define facts
friends(charles, carlos).
friends(carlos, oscal).
friends(charles, norris).


friendsoffriends(X, Y) :- friends(X,CommonPerson),
    friends(CommonPerson,Y).
