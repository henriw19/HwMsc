import pyzx as zx
import matplotlib.pyplot as plt
import stim
from stim import Circuit
from cle_zx.detection_webs import get_detection_webs, make_rg
import quizx
import numpy as np



class AltFloquetColour:
    def __init__(self, graph, Lx=0, Ly=0):
        self.size = (Lx, Ly)
        self.Lx = self.size[0]
        self.Ly = self.size[1]
        self.graph = graph
    
    def findcol(self):
        i = 0
        self.coordtocol = dict()
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                for j in range(3):
                    self.coordtocol[(x,y,j)] = i
                    i += 1
        return self.coordtocol
    
    def addzmeasurement(self, c1,c2, det:bool=False): #  c1=(x1,y1,j1), c2=(x2,y2,j2)):
        c1 = (c1[0] % self.size[0], c1[1] % self.size[1], c1[2])
        c2 = (c2[0] % self.size[0], c2[1] % self.size[1], c2[2])

        # add a z measurement between two qubits
        v = self.g.add_vertex(zx.VertexType.X, qubit=self.coordtocol[c1], row=self.t, phase=0)
        self.g.add_edge(self.g.edge(v, self.cov[c1]))

        w = self.g.add_vertex(zx.VertexType.X, qubit=self.coordtocol[c2], row=self.t, phase=0)
        self.g.add_edge(self.g.edge(w, self.cov[c2]))

        self.g.add_edge(self.g.edge(v, w))
        self.cov[c1] = v
        self.cov[c2] = w

        self.t += 1

        if det:
            self.det_edges.add((v, w))

        return self.g
    
    def addxmeasurement(self, c1,c2): #  c1=(x1,y1,j1), c2=(x2,y2,j2)):
        c1 = (c1[0] % self.size[0], c1[1] % self.size[1], c1[2])
        c2 = (c2[0] % self.size[0], c2[1] % self.size[1], c2[2])

        # add an x measurement between two qubits
        v = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[c1], row=self.t, phase=0)
        self.g.add_edge(self.g.edge(v, self.cov[c1]))

        w = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[c2], row=self.t, phase=0)
        self.g.add_edge(self.g.edge(w, self.cov[c2]))

        self.g.add_edge(self.g.edge(v, w))
        self.cov[c1] = v
        self.cov[c2] = w
        self.t += 1

        return self.g


    def zxGraph(self):

        # "row" = timestep 
        Lx = self.size[0]
        Ly = self.size[1]

        self.g = zx.Graph()
        coordtocol = self.findcol()

        self.det_edges = set() 
        #CurrentOutgoingVertex: 
        self.cov = dict()

        counter = 3 * self.Lx * self.Ly + 1
        extras = []
        for x in range(self.Lx):
            row = []
            for y in range(self.Ly):
                row.append(counter)
                counter += 1
            extras.append(row)

    # initialize all qubits:
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                v = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[(x,y,0)], row=0, phase=0)
                w = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[(x,y,0)], row=1, phase=0)

                n = self.g.add_vertex(zx.VertexType.X, qubit=self.coordtocol[(x,y,1)], row=0, phase=0)
                m = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[(x,y,1)], row=1, phase=0)
                self.g.add_edge(self.g.edge(v, w))
                self.g.add_edge(self.g.edge(n, m))
                self.g.add_edge(self.g.edge(w, m))

                self.det_edges.add((w, m))

                p = self.g.add_vertex(zx.VertexType.X, qubit=self.coordtocol[(x,y,2)], row=0, phase=0)
                self.cov[(x,y,0)] = w
                self.cov[(x,y,1)] = m
                self.cov[(x,y,2)] = p
        self.t = 2
    # add 1st round of measurements:

        for x in range(self.size[0]):
            for y in range(self.size[1]):
        
                self.addzmeasurement((x+1,y,1), (x,y-1,1))
                self.addzmeasurement((x,y,1), (x,y-1,1))
                self.addzmeasurement((x,y,1), (x+1,y,1))
                self.addzmeasurement((x+1,y,1), (x,y-1,1))
    
    # add 2nd round of measurements

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                self.addxmeasurement((x,y,1),(x,y,2))
                h = self.g.add_vertex(zx.VertexType.H_BOX, qubit = self.coordtocol[(x,y,0)], row=self.t)
                self.g.add_edge(self.g.edge(h, self.cov[(x,y,0)]))
                self.cov[(x,y,0)] = h
                self.t += 1


                v = self.g.add_vertex(zx.VertexType.X, qubit=self.coordtocol[(x,y,2)], row=self.t, phase=0)
                e = self.g.add_vertex(zx.VertexType.Z, qubit=extras[x][y], row=self.t, phase=0)
                self.g.add_edge(self.g.edge(v, e))

                h = self.g.add_vertex(zx.VertexType.H_BOX, qubit = self.coordtocol[(x,y,1)], row=self.t)
                self.g.add_edge(self.g.edge(h, self.cov[(x,y,1)]))
                self.cov[(x,y,1)] = h
                self.g.add_edge(self.g.edge(v, self.cov[(x,y,2)]))
                self.t += 1

                w = self.g.add_vertex(zx.VertexType.X, qubit=self.coordtocol[(x,y,2)], row=self.t, phase=0)
                e2 = self.g.add_vertex(zx.VertexType.Z, qubit=extras[x][y], row=self.t, phase=0)
                self.g.add_edge(self.g.edge(w, e2))
                self.g.add_edge(self.g.edge(v, w))
                self.det_edges.add((w, e2))
                self.t += 1

                n = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[(x,y,0)], row=self.t, phase=0)
                m = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[(x,y,2)], row=self.t, phase=0)
                self.g.add_edge(self.g.edge(w, m))
                
                
                self.g.add_edge(self.g.edge(self.cov[(x,y,0)], n))#, edgetype=zx.EdgeType.HADAMARD)
                self.cov[(x,y,0)] = n
                self.g.add_edge(self.g.edge(n, m))
                self.cov[(x,y,2)] = m
                

        for x in range(self.size[0]):
            for y in range(self.size[1]):

                self.addzmeasurement((x+1,y,0), (x,y-1,0), det = True)
                self.addzmeasurement((x,y,0), (x,y-1,0), det = True)
                self.addzmeasurement((x,y,0), (x+1,y,0), det = True)
                self.addzmeasurement((x+1,y,0), (x,y-1,0), det = True)
    
    # add 3rd round of measurements

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                # add 2nd round of measurements
                # self.addxmeasurement((x,y,0),(x,y,1))

                a = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[(x,y,0)], row=self.t, phase=0)
                self.g.add_edge(self.g.edge(self.cov[(x,y,0)], a))

                b = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[(x,y,1)], row=self.t, phase=0)
                h = self.g.add_vertex(zx.VertexType.H_BOX, qubit = self.coordtocol[(x,y,2)], row=self.t)
                self.g.add_edge(self.g.edge(h, self.cov[(x,y,2)]))
                self.cov[(x,y,2)] = h
                
                self.g.add_edge(self.g.edge(self.cov[(x,y,1)], b))#,edgetype=zx.EdgeType.HADAMARD)
                self.g.add_edge(self.g.edge(a, b))
                self.cov[(x,y,0)] = a


                self.t += 1

                v = self.g.add_vertex(zx.VertexType.X, qubit=self.coordtocol[(x,y,1)], row=self.t, phase=0)
                e = self.g.add_vertex(zx.VertexType.Z, qubit=extras[x][y], row=self.t, phase=0)
                self.g.add_edge(self.g.edge(v, e))
                self.det_edges.add((v, e))


                h = self.g.add_vertex(zx.VertexType.H_BOX, qubit = self.coordtocol[(x,y,0)], row=self.t)
                self.g.add_edge(self.g.edge(h, self.cov[(x,y,0)]))
                self.cov[(x,y,0)] = h
                self.g.add_edge(self.g.edge(v, b))
                self.t += 1


                w = self.g.add_vertex(zx.VertexType.X, qubit=self.coordtocol[(x,y,1)], row=self.t, phase=0)
                e2 = self.g.add_vertex(zx.VertexType.Z, qubit=extras[x][y], row=self.t, phase=0)
                self.g.add_edge(self.g.edge(w, e2))
                self.g.add_edge(self.g.edge(v, w))

                self.t += 1
                n = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[(x,y,1)], row=self.t, phase=0)
                m = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[(x,y,2)], row=self.t, phase=0)
                self.g.add_edge(self.g.edge(w, n))
                self.g.add_edge(self.g.edge(n, m))
                self.g.add_edge(self.g.edge(self.cov[(x,y,2)], m))#, edgetype=zx.EdgeType.HADAMARD)   
                self.cov[(x,y,1)] = n
                self.cov[(x,y,2)] = m  

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                self.addzmeasurement((x+1,y,2), (x,y-1,2))
                self.addzmeasurement((x,y,2), (x,y-1,2))
                self.addzmeasurement((x,y,2), (x+1,y,2))
                self.addzmeasurement((x+1,y,2), (x,y-1,2))
    
    # finalize all qubits
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                # self.addxmeasurement((x,y,2),(x,y,0))
                a = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[(x,y,2)], row=self.t, phase=0)
                self.g.add_edge(self.g.edge(self.cov[(x,y,2)], a))

                b = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[(x,y,0)], row=self.t, phase=0)
                self.g.add_edge(self.g.edge(self.cov[(x,y,0)], b))#,edgetype=zx.EdgeType.HADAMARD)
                self.g.add_edge(self.g.edge(a, b))
                self.cov[(x,y,2)] = a
                self.t += 1

                self.det_edges.add((a, b))
            


                v = self.g.add_vertex(zx.VertexType.Z, qubit=self.coordtocol[(x,y,0)], row=self.t, phase=0)
                self.g.add_edge(self.g.edge(v, b))

                n = self.g.add_vertex(zx.VertexType.X, qubit=self.coordtocol[(x,y,1)], row=self.t, phase=0)
                self.g.add_edge(self.g.edge(n, self.cov[(x,y,1)]))

                m = self.g.add_vertex(zx.VertexType.X, qubit=self.coordtocol[(x,y,2)], row=self.t, phase=0)
                self.g.add_edge(self.g.edge(m, self.cov[(x,y,2)]))
        self.g = make_rg(self.g)
        return self.g
    
    def detectorFinder(self):
        greenWebs = set()
        redWebs = set()

        g = self.zxGraph()
        g = make_rg(g)
        pauliWebs = get_detection_webs(g)
        for i,web in enumerate(pauliWebs):
            vertices = sorted(web.vertices())
            anchor = list(vertices)[0]
            # print(list(vertices)[0])
            neighbors = set(g.neighbors(list(vertices)[0]))
            if len(neighbors & set(vertices))%2 == 0:
                if g.type(anchor) == zx.VertexType.X:
                    redWebs.add(web)
                else:
                    greenWebs.add(web)
            else:
                if g.type(anchor) == zx.VertexType.X:
                    greenWebs.add(web)
                else:
                    redWebs.add(web)

        print("found:",len(pauliWebs), len(greenWebs), len(redWebs))
        print(greenWebs)

                            
                    
                

    def stimCircuit(self, detectors:bool=True):
        self.ix= 0
        self.iz = 0
        self.ih = 0
        ancilla = 4*self.Lx*self.Ly + 1
        
        self.m = 0
        # total # of measurements: Lx*Ly*25

        
        # circuit += Circuit(f"DETECTOR {' '.join([f'rec[{d[l]}]' for l in range(len(d))])}") 

        def measurex(target_qubits):  #=doppel grün
            self.ix += 1
            # print("measurex", target_qubits)
            self.circ += Circuit(f'''
                                H {ancilla}
                                CX {' '.join(f'{ancilla} {x}' for x in target_qubits)}
                                H {ancilla}
                                M {ancilla}
                                R {ancilla}
                                ''')
    
        def measurez(target_qubits): #doppel rot
            self.iz += 1
            # print("measurez", target_qubits)
            self.circ += Circuit(f'''
                                CX {' '.join(f'{x} {ancilla}' for x in target_qubits)}
                                M {ancilla}
                                R {ancilla}
                                ''')
        

        # Convert the ZX graph to a stim circuit
        g = self.zxGraph()
        self.circ = Circuit()
        mm =0
        measured = set()  # Keep track of measured vertices
        for edge in g.edges():
            if g.row(edge[0]) == 0 and g.type(edge[0]) == zx.VertexType.Z:
                    # Handle Z inputs
                    q = g.qubit(edge[0])
                    self.circ += Circuit(f'H {q}')

            if g.row(edge[0]) == g.row(edge[1]) and edge[1] not in measured:
                # Handle intermediate measurements
                if len(g.neighbors(edge[1])) == 2:
                    q1 = list(g.neighbors(edge[1]))[0]
                    q2 = list(g.neighbors(edge[1]))[1]
                    if g.type(edge[1]) == zx.VertexType.Z:
                        measured.add(edge[1])
                        measurez((q1, q2))
                    elif g.type(edge[1]) == zx.VertexType.X:
                        measured.add(edge[1])
                        measurex((q1, q2))
                if len(g.neighbors(edge[1])) == 1:
                    if g.type(edge[1]) == zx.VertexType.Z:
                        measurez([g.qubit(edge[0])])

            if g.type(edge[0]) == zx.VertexType.H_BOX:
                # Handle Hadamard boxes
                self.ih += 1
                q = g.qubit(edge[0]) 
                self.circ += Circuit(f'H {q}')

            if g.row(edge[0]) in set(range(g.depth(), g.depth() - self.Lx*self.Ly, -1)) and len(g.neighbors(edge[0])) == 1:
                mm += 1
                if g.type(edge[0]) == zx.VertexType.Z: #final measurements
                    q = g.qubit(edge[0])
                    measurex([q])
                elif g.type(edge[0]) == zx.VertexType.X:
                    q = g.qubit(edge[0])
                    measurez([q])

        
        print("Circuit generated with", self.ix, "X measurements", self.iz, "Z measurements", self.ih, "Hadamards.")
        return self.circ
    
    def quizx(self):
        g = zx.Graph()
        circ = quizx.zx2stim(l)
        print(circ)




mycol = AltFloquetColour(graph=None, Lx=2, Ly=2) #2x2: 216, 3x2: 324, 3x3:486
# mycol.quizx()
g = mycol.zxGraph()
det = mycol.detectorFinder()
# print(det)
# circ = mycol.stimCircuit()
fig = zx.draw(g, labels=True)
# fig.savefig("zx_diagram.png", dpi=1000, bbox_inches='tight')
# print(len(g.neighbors(55)))




