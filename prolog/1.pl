%prolog program to implement friend of friend
%!  defining rules facts
friends(ram,hari).
friends(hari,ravan).
friends(ram,shyam).

%defining rules
friendsoffriends(X,Y):-
friends(X,CommonFriend),
friends(CommonFriend,Y).
