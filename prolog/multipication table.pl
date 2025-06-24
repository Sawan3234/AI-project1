mulTable(N,End):-  mulTable(N,1,End).
mulTable(_ ,Start,End):-   Start > End.

mulTable(N,Counter, End):-  Counter=<End,  write(N),  write('X'),  write(Counter), write('='),  Mul is N * Counter , write(Mul), nl , NewCounter is Counter +1,
    mulTable(N,NewCounter,End).
