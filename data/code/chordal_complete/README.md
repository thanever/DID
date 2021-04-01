# README #

This is an implementation to solve the Minimum Fill In problem (called MFI), done in the context of the PACE 2017 challenge
https://pacechallenge.wordpress.com/pace-2017/track-b-minimum-fill-in/

A description of the algorithm used can be found in the document [description.pdf](description.pdf).

To program wait for a description of the graph in stdin in the form "one line = one edge" where an edge is the two ids of each node.
It gives in output the solution, i.e. the edges to remove to make the graph chordal.

Typical use: cat instance.graph | python main.py

### Authors ###
Édouard Bonnet  
Middlesex University, UK

R. B. Sandeep  
Institute for Computer Science and Control  
Hungarian Academy of Sciences (MTA SZTAKI), Hungary

Florian Sikora  
LAMSADE, CNRS, Université Paris-Dauphine, France

### Requirement: ###
Python 2
Networkx

### License ###
This project is under the GPLv3 license (see [license](LICENSE))
