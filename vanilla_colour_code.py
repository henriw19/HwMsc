# from hwbsc.hexcolor_CLEMENS import make_a_base_graph, draw_graph_with_colored_edges_and_nodes, find_6_loops
# from hwbsc.almlinunionfind import Alm_lin_union_find
import networkx as nx
import matplotlib.pyplot as plt
# from hwbsc.stab_codes import StabCode
import numpy as np
# from hwbsc.stabilizers_utils import find_log_op_color
from ldpc.mod2 import rank, nullspace, row_basis, row_span, row_echelon, row_basis
from ldpc.code_util import compute_code_distance
import stim
from pathlib import Path

# G = make_a_base_graph(6,4)

# nx.draw(G)
# plt.show()
# draw_graph_with_colored_edges_and_nodes(G)




# def ishit(G)->bool:
#     for node in G.nodes:
#             redcounter = 0
#             greencounter = 0
#             bluecounter = 0
#             for edge in G.edges(node):
#                 #print(edge)
#                 if G.edges[edge]['color'] == 'g':
#                     greencounter += 1
#                 elif G.edges[edge]['color'] == 'b':
#                     bluecounter += 1
#                 elif G.edges[edge]['color'] == 'r':
#                     redcounter += 1
#             if redcounter > 1 or greencounter > 1 or bluecounter > 1:
#                 return True

# for i in range(30):
#     for j in range(i):
      
#         G = make_a_base_graph(2*i,2*j)
#         if ishit(G):
#             print(2*i,2*j)

def hexa_color_pcm_generator(m,n):
    """
    n is number of horizontal faces
    m is number of vertical faces
    m must be a int divisible by 6
    n must be smaller than m and even
    """
    
    if m%6 != 0:
        raise ValueError(m,"m must be a int divisible by 6")
    
    if n%2 != 0 or n>= m:
        raise ValueError(n,"n must be smaller than m and even")
    
    num_faces = n*m
    num_nodes = 2* num_faces
    facelist = []
    pcm = []
    for horface in range(n):
        offset = 0
        offset2 = 0
        if horface%2 != 0:
            offset = n
        
        if horface%(n-1) == 0 and horface != 0:
            offset2 = -n

        for verface in range(m):
            offset3 = 0
            if verface%(m-1) == 0 and verface != 0:
                offset3 = -2*(m * n)
            
            stab = np.zeros(6)
            stab[0] = 2*n*verface + horface + offset        
            stab[1] = stab[0] + 1 + offset2  
            if horface%2 != 0:
                stab[2] = stab[0] + n + offset3                    
                stab[3] = stab[0] + n + 1 + offset2 + offset3
            else:
                stab[2] = stab[0] + n                    
                stab[3] = stab[0] + n + 1 + offset2 
            stab[4] = stab[0] + 2*n + offset3
            stab[5] = stab[0] + 2*n + 1 + offset2 + offset3

            binstab = np.zeros(num_nodes).astype(int)
            binstab[stab.astype(int)] = 1
            
            facelist.append(stab[0].astype(int))
            pcm.append(binstab)    
    return np.array(pcm), facelist

# import panqec.codes

def tri_color_pcm_generator(x,y, plot = False):
    code = panqec.codes.color_2d.Color666PlanarCode(x,y)
    
    n = code.n
    k = code.k
    d = code.d
    mat = code.stabilizer_matrix.toarray()[1::2,n:]

    stabmatrix = np.block([[np.zeros(mat.shape),mat],[mat,np.zeros(mat.shape)]])
    # print(StabCode(stabmatrix,distance=True).d)

    if plot == True:
        edge_matrix = np.block([[np.zeros((mat.shape[0],mat.shape[0]),dtype=int),mat],
                                [mat.T,np.zeros((mat.shape[1],mat.shape[1]),dtype=int)]])
        triG = nx.from_numpy_array(edge_matrix)
        nx.draw(triG, with_labels = True)
        # plt.show()
    return mat, code, n, k, d

def panqec_hexa_color_pcm_generator(x,y, plot = False):
    code = panqec.codes.color_2d.Color666ToricCode(x,y)
    
    n = code.n
    k = code.k
    d = code.d
    mat = code.stabilizer_matrix.toarray()[1::2,n:]
    print("panqec toric pcm shape",mat.shape)
    stabmatrix = np.block([[np.zeros(mat.shape),mat],[mat,np.zeros(mat.shape)]])
    # print(StabCode(stabmatrix,distance=True).d)

    if plot == True:
        edge_matrix = np.block([[np.zeros((mat.shape[0],mat.shape[0]),dtype=int),mat],
                                [mat.T,np.zeros((mat.shape[1],mat.shape[1]),dtype=int)]])
        triG = nx.from_numpy_array(edge_matrix)
        nx.draw(triG, with_labels = True)
        # plt.show()
    return mat, code, n, k, d
  

def hexa_color_graph(pcm: np.ndarray):
    mat = pcm
    edge_matrix = np.block([[np.zeros((mat.shape[0],mat.shape[0]),dtype=int),mat],
                            [mat.T,np.zeros((mat.shape[1],mat.shape[1]),dtype=int)]])
    # edge_matrix = np.block([[np.zeros((mat.shape[1],mat.shape[1]),dtype=int),mat.T],
    #                         [mat,np.zeros((mat.shape[0],mat.shape[0]),dtype=int)]])
    


    colG = nx.from_numpy_array(edge_matrix)
    faces = []
    print("stop1")
    for i, node in enumerate(edge_matrix):
        if len(list(colG.neighbors(i))) == 6:
            faces.append(i)
    # print(faces)
    
    
    
    

    distance = nx.shortest_path_length(colG, source=faces[4], target=faces[5])
    # initialize all face colors
    nx.set_node_attributes(colG, {n: '0' for n in faces}, name='color')
    colG.nodes[faces[0]]['color'] = 'r'
    colG.nodes[faces[1]]['color'] = 'g'

    colors = {'r','g','b'}

    print("stop2")
    for face in faces:
        closest_neighbor_faces = []
        if colG.nodes[face]['color'] != '0':
            for face2 in faces:
                if colG.nodes[face2]['color'] != '0':
                    if face != face2 and nx.shortest_path_length(colG, source=face, target=face2) == 2:
                        for face3 in faces:
                            if nx.shortest_path_length(colG, source=face, target=face3) == 2:
                                if nx.shortest_path_length(colG, source=face2, target=face3) == 2:
                                    if face != face3 and face2 != face3:
                                        newcolor = next(iter(colors.difference({colG.nodes[face]['color'], colG.nodes[face2]['color']})))
                                        colG.nodes[face3]['color'] = newcolor


    print("stop3")
    # print(faces)
    # print(faces)
    facegraph = colG.subgraph(faces).copy()
    print("stop4")
    # add edges to facegraph
    edges = []
    for face in faces:
        for neighbor in faces:
            if face != neighbor:
                #print("oben",face,neighbor)
                if nx.shortest_path_length(colG, source=face, target=neighbor) == 2:
                    #print("unten",face,neighbor)
                    facegraph.add_edge(face, neighbor)
                    edgecolor = next(iter(colors.difference({facegraph.nodes[face]['color'], facegraph.nodes[neighbor]['color']})))
                    facegraph.edges[(face,neighbor)]['color'] = edgecolor

    print("stop5")
    node_colors = [data['color'] for _, data in facegraph.nodes(data=True)] 
    edge_colors = [data['color'] for _,_, data in facegraph.edges(data=True)] 



    # nx.draw(colG, with_labels = True, node_color = data_color)
    # nx.draw(facegraph, node_color=node_colors, edge_color=edge_colors, with_labels = True)
    # plt.show()
    # print(facegraph.nodes(data=True))

    #translation dictionary between colG and facegraph:
    transl = {}
    for node in colG.nodes():
        if colG.nodes[node].get('color') != None:
            pass
            # print(node)

    print("stop6")
    # for face in colG:
    #     if face in faces:           
    #         rc = 0
    #         gc = 0
    #         bc = 0
    #         c=0
    #         for neighbor in colG:
    #             if nx.shortest_path_length(colG, source=face, target=neighbor) < 3:
    #                 if neighbor in faces and face != neighbor:
    #                     c += 1
    #                     # print(face,neighbor)
    #                     if colG.nodes[neighbor]['color'] == 'r':
    #                         rc += 1
    #                     if colG.nodes[neighbor]['color'] == 'g':
    #                         gc += 1
    #                     if colG.nodes[neighbor]['color'] == 'b':
    #                         bc += 1
    #         if c!= 6:
    #             print("error")

    print("stop7")
    # check if legit
    # for node in facegraph.nodes:
    #     colorsdic = {'r':0,'g':0,'b':0}
    #     for neighbor in facegraph.neighbors(node):
    #         #print(neighbor,"\n")
    #         othercolors = []
    #         if facegraph.nodes[neighbor]['color'] == 'g':
    #             colorsdic['g'] += 1
    #         elif facegraph.nodes[neighbor]['color'] == 'b':
    #             colorsdic['b'] += 1
    #         elif facegraph.nodes[neighbor]['color'] == 'r':
    #             colorsdic['r'] += 1
    #     del colorsdic[facegraph.nodes[node]['color']]
    #     keys = list(colorsdic.keys())
    #     if not colorsdic[keys[0]] == 3 and colorsdic[keys[1]] == 3:
    #         raise ValueError("Failed to color graph")

    print("stop8")
    # 2nd check if legit
    # for node in list(colG.nodes()):
    #     if node in list(facegraph.nodes()):
    #         if len(list(colG.neighbors(node))) != 6:
    #             raise ValueError("Failed to build graph")
    #     if node not in list(facegraph.nodes()):
    #         if len(list(colG.neighbors(node))) != 3:
    #             raise ValueError("Failed to build graph")

    return colG, facegraph, node_colors, edge_colors


def build_toric_colour_code_circuit_from_pcm(Lx: int, Ly: int, rounds: int) -> stim.Circuit:
    """Build a Stim circuit for the toric 6.6.6 colour code using the PCM generator.

    Plaquettes are 3-coloured via graph colouring. Z-stabilisers are measured first
    (data -> ancilla CX), then X-stabilisers (H ancilla, ancilla -> data CX, H),
    grouped by colour so no data qubit is touched twice in a sub-round.
    """
    m = 6 * Lx
    n = 6 * Ly

    pcm, facelist = hexa_color_pcm_generator(m, n)
    num_checks, num_data = pcm.shape
    ancilla_offset = num_data

    # Build adjacency between plaquettes sharing data qubits and 3-colour it.
    supports = [set(np.nonzero(pcm[i])[0]) for i in range(num_checks)]
    for i, supp in enumerate(supports):
        if len(supp) != 6:
            raise ValueError(f"Plaquette {i} has {len(supp)} adjacent qubits (expected 6).")
    adj = nx.Graph()
    adj.add_nodes_from(range(num_checks))
    for data_idx in range(num_data):
        faces_with = [i for i, s in enumerate(supports) if data_idx in s]
        for a in range(len(faces_with)):
            for b in range(a + 1, len(faces_with)):
                adj.add_edge(faces_with[a], faces_with[b])
    colouring = nx.coloring.greedy_color(adj, strategy="DSATUR")
    max_color = max(colouring.values()) if colouring else -1
    if max_color > 2:
        raise ValueError(f"Failed to 3-colour plaquettes; need <=3 colours, got {max_color + 1}")

    colour_groups = {0: [], 1: [], 2: []}
    for idx in range(num_checks):
        c = colouring.get(idx, 0)
        colour_groups[c].append(idx)

    # Observable: straight line of six data qubits (first six along data index order)
    logical_line = list(range(min(6, num_data)))
    if len(logical_line) >= 2:
        logical_pairs = [(logical_line[i], logical_line[(i + 1) % len(logical_line)]) for i in range(len(logical_line))]
    else:
        logical_pairs = []

    circuit = stim.Circuit()
    data_targets = list(range(num_data))
    if data_targets:
        circuit.append("R", data_targets)

    measurement_count = 0
    last_meas = {"Z": {}, "X": {}}

    for r in range(rounds):
        # Z stabilisers: data -> ancilla CX, measure ancilla.
        for colour in (0, 1, 2):
            for idx in colour_groups[colour]:
                anc = ancilla_offset + idx
                circuit.append("R", [anc])
                data_idxs = list(np.nonzero(pcm[idx])[0])
                for d in data_idxs:
                    circuit.append("CX", [d, anc])
                circuit.append("M", [anc])
                measurement_count += 1
                prev = last_meas["Z"].get(idx)
                if prev is not None:
                    delta = measurement_count - prev
                    circuit.append(
                        "DETECTOR",
                        [stim.target_rec(-1), stim.target_rec(-(delta + 1))],
                    )
                last_meas["Z"][idx] = measurement_count
        # X stabilisers: prepare ancilla in |+>, ancilla -> data CX, measure in X.
        for colour in (0, 1, 2):
            for idx in colour_groups[colour]:
                anc = ancilla_offset + idx
                circuit.append("R", [anc])
                circuit.append("H", [anc])
                data_idxs = list(np.nonzero(pcm[idx])[0])
                for d in data_idxs:
                    # Use data as control so logical Z operators stay on data.
                    circuit.append("CX", [d, anc])
                circuit.append("H", [anc])
                circuit.append("M", [anc])
                measurement_count += 1
                prev = last_meas["X"].get(idx)
                if prev is not None:
                    delta = measurement_count - prev
                    circuit.append(
                        "DETECTOR",
                        [stim.target_rec(-1), stim.target_rec(-(delta + 1))],
                    )
                last_meas["X"][idx] = measurement_count

    # Extra terminal Z-round so the last operations touching data are Z-type,
    # making the final destructive Z readout deterministic.
    for colour in (0, 1, 2):
        for idx in colour_groups[colour]:
            anc = ancilla_offset + idx
            circuit.append("R", [anc])
            data_idxs = list(np.nonzero(pcm[idx])[0])
            for d in data_idxs:
                circuit.append("CX", [d, anc])
            circuit.append("M", [anc])
            measurement_count += 1
            prev = last_meas["Z"].get(idx)
            if prev is not None:
                delta = measurement_count - prev
                circuit.append(
                    "DETECTOR",
                    [stim.target_rec(-1), stim.target_rec(-(delta + 1))],
                )
            last_meas["Z"][idx] = measurement_count

    # Final destructive Z-measurement of all data qubits.
    if data_targets:
        circuit.append("M", data_targets)
        # Total number of recorded measurements (ancilla + final data).
        total_meas = measurement_count + len(data_targets)
        # Include the data-qubit Z readout along the logical line.
        for q in logical_line:
            offset = len(data_targets) - q
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-offset)], [0])
        # Also include the last-round X-stabiliser outcomes touching the logical line
        # so the logical observable commutes with earlier ancilla resets.
        logical_ancillas = {
            idx
            for idx in range(num_checks)
            if any(pcm[idx, d] for d in logical_line)
        }
        for idx in logical_ancillas:
            m_idx = last_meas["X"].get(idx)
            if m_idx is None:
                continue
            offset = total_meas - m_idx + 1
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-offset)], [0])

    return circuit


def tri_color_graph(pcm: np.ndarray):
    mat = pcm
    edge_matrix = np.block([[np.zeros((mat.shape[0],mat.shape[0]),dtype=int),mat],
                            [mat.T,np.zeros((mat.shape[1],mat.shape[1]),dtype=int)]])
    # edge_matrix = np.block([[np.zeros((mat.shape[1],mat.shape[1]),dtype=int),mat.T],
    #                         [mat,np.zeros((mat.shape[0],mat.shape[0]),dtype=int)]])
    


    colG = nx.from_numpy_array(edge_matrix)
    faces = np.arange(pcm.shape[0])
        

    # distance = nx.shortest_path_length(colG, source=faces[4], target=faces[5])
    # initialize all face colors
    nx.set_node_attributes(colG, {n: '0' for n in faces}, name='color')
    colG.nodes[faces[0]]['color'] = 'r'
    colG.nodes[faces[1]]['color'] = 'g'

    colors = {'r','g','b'}


    for face in faces:
        closest_neighbor_faces = []
        if colG.nodes[face]['color'] != '0':
            for face2 in faces:
                if face != face2 and nx.shortest_path_length(colG, source=face, target=face2) == 2 and colG.nodes[face2]['color'] != '0':
                    for face3 in faces:
                        if nx.shortest_path_length(colG, source=face, target=face3) == 2  and nx.shortest_path_length(colG, source=face2, target=face3) == 2:
                            if face != face3 and face2 != face3:
                                newcolor = next(iter(colors.difference({colG.nodes[face]['color'], colG.nodes[face2]['color']})))
                                colG.nodes[face3]['color'] = newcolor

    facegraph = colG.subgraph(faces).copy()

    # add edges to facegraph
    edges = []
    for face in faces:
        for neighbor in faces:
            if face != neighbor:
                #print("oben",face,neighbor)
                if nx.shortest_path_length(colG, source=face, target=neighbor) == 2:
                    #print("unten",face,neighbor)
                    facegraph.add_edge(face, neighbor)
                    edgecolor = next(iter(colors.difference({facegraph.nodes[face]['color'], facegraph.nodes[neighbor]['color']})))
                    facegraph.edges[(face,neighbor)]['color'] = edgecolor

  
    node_colors = [data['color'] for _, data in facegraph.nodes(data=True)] 
    edge_colors = [data['color'] for _,_, data in facegraph.edges(data=True)] 



    nx.draw(colG, with_labels = True)#, node_color = data_color)
    # nx.draw(facegraph, node_color=node_colors, edge_color=edge_colors, with_labels = True)
    plt.show()
    # print(facegraph.nodes(data=True))

    #translation dictionary between colG and facegraph:
    transl = {}
    for node in colG.nodes():
        if colG.nodes[node].get('color') != None:
            pass
            # print(node)

# no check:
    # for face in colG:
    #     if face in faces:           
    #         rc = 0
    #         gc = 0
    #         bc = 0
    #         c=0
    #         for neighbor in colG:
    #             if nx.shortest_path_length(colG, source=face, target=neighbor) < 3:
    #                 if neighbor in faces and face != neighbor:
    #                     c += 1
    #                     # print(face,neighbor)
    #                     if colG.nodes[neighbor]['color'] == 'r':
    #                         rc += 1
    #                     if colG.nodes[neighbor]['color'] == 'g':
    #                         gc += 1
    #                     if colG.nodes[neighbor]['color'] == 'b':
    #                         bc += 1
    #         if c!= 6:
    #             pass
    #             # print("error")

    # check if legit    NOOOOOOOOOOOOO
    # for node in facegraph.nodes:
    #     colorsdic = {'r':0,'g':0,'b':0}
    #     for neighbor in facegraph.neighbors(node):
    #         #print(neighbor,"\n")
    #         othercolors = []
    #         if facegraph.nodes[neighbor]['color'] == 'g':
    #             colorsdic['g'] += 1
    #         elif facegraph.nodes[neighbor]['color'] == 'b':
    #             colorsdic['b'] += 1
    #         elif facegraph.nodes[neighbor]['color'] == 'r':
    #             colorsdic['r'] += 1
    #     del colorsdic[facegraph.nodes[node]['color']]
    #     keys = list(colorsdic.keys())
    #     if not colorsdic[keys[0]] == 3 and colorsdic[keys[1]] == 3:
    #         pass
    #         # raise ValueError("Failed to color graph")

    # # 2nd check if legit
    # for node in list(colG.nodes()):
    #     if node in list(facegraph.nodes()):
    #         if len(list(colG.neighbors(node))) != 6:
    #             pass
    #             # raise ValueError("Failed to build graph")
    #     if node not in list(facegraph.nodes()):
    #         if len(list(colG.neighbors(node))) != 3:
    #             pass
    #             # raise ValueError("Failed to build graph")

    return colG, facegraph, node_colors, edge_colors



def color_code_decoder(syndrome, color_graph, facegraph):

    blue_edges = []
    for edge in facegraph.edges:
        if facegraph.edges[edge]['color'] == 'b':
            blue_edges.append(edge)

    

    blue_subgraph = facegraph.edge_subgraph(blue_edges)
    subpcm = nx.to_numpy_array(blue_subgraph).astype(int)
    subpcm = subpcm.T
    
    for i,row in enumerate(subpcm):
        
        if sum(row) != 3:
            print(row)
            subpcm = np.delete(subpcm, i,axis= 0)
    
    bluepcm = nx.to_numpy_array(blue_subgraph)
    bluee = nx.from_numpy_array(bluepcm)
    
    # print(list(bluee.edges))
    # bluee.remove_edges_from()
    

    translation = {}
    for i,node in enumerate(blue_subgraph.nodes()):
        translation[i] = node

    num_ancillas = blue_subgraph.erer_of_nodes()
    num_edges = blue_subgraph.number_of_edges()
    subpcm2 = np.zeros((num_ancillas, num_edges)).astype(int)

    
    edge_indices = {index:edge for index, edge in enumerate(blue_subgraph.edges())}
    c = 0
    for ancilla in range(num_ancillas):
        for i,edge in enumerate(blue_edges):
            if translation[ancilla] in edge:
                subpcm2[ancilla][i] = 1
                print("zeile:",ancilla)
                print(translation[ancilla])
                
                print(i,edge)
                print(subpcm2[ancilla])
                print("RANK",rank(subpcm2))
    


                # if rank(subpcm2) == 16:
                #     c += 1
                # if c == 2:
                #     break
                # print(edge)
                # print(rank(subpcm2),ancilla)
                # print("added edge:",edge)
            

    
    
    subsyndrome = np.zeros(num_ancillas, dtype=int)
    # subsyndrome = syndrome[blue_subgraph.nodes()]

    mat = subpcm2
    edge_matrix = np.block([[np.zeros((mat.shape[0],mat.shape[0]),dtype=int),mat],
                            [mat.T,np.zeros((mat.shape[1],mat.shape[1]),dtype=int)]])
    G = nx.from_numpy_array(edge_matrix)
    
    for node in G.nodes:
        nb = list(G.neighbors(node))
        if len(nb) == 2:
            pass
    
    
    vec = np.array([1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0])
    print(bluepcm.astype(int) @ vec %2)
                
                        
                            
    
    anccount = 0
    fakedata = 0
    for node in G.nodes():
        if len(list(G.neighbors(node))) == 3:
            anccount += 1
        elif len(list(G.neighbors(node))) == 2:
            fakedata += 1
        else:
            # print("fail")
            pass

    
        
    # print(anccount,fakedata)
    # nx.draw(bluee, with_labels=True)
    
    # plt.show()
    # decodepred = gnversion(subpcm2,subsyndrome)
    # return decodepred



# pcm,code,n,k,d = panqec_hexa_color_pcm_generator(8,10)
# print(n,k,d)

# import sys
# pcm,code,n,k,d = tri_color_pcm_generator(15,15)
# logical = code.logicals_x
# half = len(logical[0])//2
# # Open a file in write mode to save the #print output
# with open('planar15logical.txt', 'w') as f:
#     # Redirect the standard output to the file
#     sys.stdout = f

#     # Your #print statements here
#     print(list(logical[0][:half]))
#     # print(pcm.tolist())

    

# Reset the standard output back to its original value (console)
# sys.stdout = sys.__stdout__



# print(hexa_color_pcm_generator(30,24)[0].shape)

# print(n,k,d)
# colG, facegraph, node_colors, edge_colors = tri_color_graph(pcm)
# # logical = code.logicals_x
# half = len(logical[0])//2
# print(list(logical[0][:half]))



# guessedlog = [0,0,1,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0]

# print(pcm)
# nx.draw(colG, node_color = node_colors, edge_color = edge_colors)




# kern = nullspace(pcm)
# logicals = []
# lamda = row_span(pcm)
# for potential_log in kern:
#     temp = np.vstack((lamda,potential_log))
#     if rank(temp) > rank(lamda):
#         lamda = temp
#         logicals.append(potential_log)
# logicals = row_basis(row_span(np.array(logicals)))
# for log in logicals:
#     print(sum(log))
#     print(log)


def draw_toric_colour_grid(Lx: int = 2, Ly: int = 1):
    """Quick visual check of the plaquette 3-colouring on an Lx×Ly torus."""
    m, n = 6 * Lx, 6 * Ly
    pcm, _ = hexa_color_pcm_generator(m, n)
    num_checks = pcm.shape[0]
    supports = [set(np.nonzero(pcm[i])[0]) for i in range(num_checks)]
    adj = nx.Graph()
    adj.add_nodes_from(range(num_checks))
    for data_idx in range(pcm.shape[1]):
        faces = [i for i, s in enumerate(supports) if data_idx in s]
        for a in range(len(faces)):
            for b in range(a + 1, len(faces)):
                adj.add_edge(faces[a], faces[b])
    colouring = nx.coloring.greedy_color(adj, strategy="DSATUR")
    colors = ['red', 'green', 'blue']
    pos = {r: (r % m, r // m) for r in range(num_checks)}
    node_cols = [colors[colouring[i] % 3] for i in adj.nodes()]
    nx.draw(adj, pos=pos, node_color=node_cols, node_size=200, with_labels=False)
    plt.gca().invert_yaxis()
    plt.title(f"Toric 6.6.6 colouring, Lx={Lx}, Ly={Ly}")
    plt.show()


if __name__ == "__main__":
    Lx= 18
    Ly= 9
    circ = build_toric_colour_code_circuit_from_pcm(Lx, Ly, 4)
    Path(f"vanilla_toric_circuit{Lx}x{Ly}.txt").write_text(str(circ))
    print("Wrote vanilla_toric_circuit.txt (Lx=6, Ly=3, rounds=4)")
