% Facts defining links (edges) between nodes
link(bahrain,saudi).
link(bahrain,japan).
link(saudi,australia).
link(saudi,china).
link(australia,miami).
link(china,miami).
link(japan,china).
path(Startcircuit,Startcircuit):-true.
path(Startcircuit,Endcircuit):-link(Startcircuit,Nextcircuit),
    path(Nextcircuit,Endcircuit).
