import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sb
import matplotlib.pyplot as plt
import itertools


# --------------------------------------------------------- getting the synergy information and the graph ----------------------------------

def parse_synergies():
    synergies = pd.read_csv('tft_synergies_set4.csv', header = None)[0]
    
    # parsing csv file into dataframe
    synergies_df = {}
    i = 0
    while i < len(synergies):
        name = synergies[i]
    #     print('name is ', name)
        syns = []
        while (not synergies[i+1].isnumeric()):
            i = i+1
            syns.append(synergies[i])
        i = i+1
        cost = synergies[i]

        synergies_df[name] = (syns, cost)

        i = i+1

    return(synergies_df)

def get_all_synergies(synergies_df):
    all_synergies = [i[0] for i in list(synergies_df.values())]
    all_synergies = np.unique([i for s in all_synergies for i in s])
    return(all_synergies)


def get_synergy_to_node(all_synergies, synergies_df, nodes):
    synergy_to_node = {s:[] for s in all_synergies}
    for n in nodes:
        for s in synergies_df[n][0]:
            synergy_to_node[s].append(n)

    return(synergy_to_node)

def tuple_equal(t1, t2):
    return(set(t1) == set(t2))

def get_edges(synergies_df, synergy_to_node, nodes):
    # now go through all nodes and make edges
    edges = []
    for n in nodes:
        for s in synergies_df[n][0]: # for each of the node's synergies
            # add an edge to everyone else in the synergy
            to_add = [(n, other, s) for other in synergy_to_node[s]]
            edges.extend(to_add)
            
    unique_edges = [edges[1]] # don't add 0 as it maps to itself (Ahri, Ahri)
    for e in edges[2:]: 
        if (not (np.sum([tuple_equal(e, i) for i in unique_edges]) > 0)) and (e[0] != e[1]): # can't already be in unique edges and can't map to itself
            unique_edges.append(e)
        
    return(edges, unique_edges)

def get_graph(nodes, unique_edges, all_synergies):
    G = nx.Graph()

    for node in nodes:
        G.add_node(node)

    edge_labels = []
    for edge in unique_edges:
        G.add_edge(edge[0], edge[1], label=edge[2], group = edge[2])
        edge_labels.append(edge[2])
    
    # edge colours
    edge_col = []
    colours= sb.color_palette('Pastel2') + sb.color_palette('Paired') + sb.color_palette('Accent')
    colours = set(colours) # removing duplicates
    col_map = {s:c for s, c in zip(all_synergies, colours)}
    for u, v in G.edges():
        s = G[u][v]['label']  
        edge_col.append(col_map[s]) # add the colour of the synergy

    return(G, edge_labels, edge_col, col_map)


def get_synergy_info():
    # now reading in synergy info, got from https://lolchess.gg/synergies
    synergy_info = pd.read_csv('tft_synergy_info_set4.csv', header = None)[0]
    
    synergy_info_dict = {}
    i = 0
    while i < len(synergy_info):
        synergy = synergy_info[i]
        i = i + 2 # skip the cost line and go onto the next line
        if synergy_info[i][0] != '(': # some have info before the synergy number infomation
            info = synergy_info[i]
            i = i + 1
        else:
            info = ''
        synergy_number_info = []
        while ((i < len(synergy_info)) and (synergy_info[i][0] == '(')):
            synergy_number_info.append((int(synergy_info[i][1]), synergy_info[i][4:])) # appending the synergy number and the information in a tuple. The form is always '(2) Blah Blah...'
            i = i+1
        synergy_info_dict[synergy] = {'info': info, 'synergy_number_info': synergy_number_info}


    # now making super simple dict, synergy to number of that synergy that has an effect
    synergy_to_numbers = {}
    for k, v in synergy_info_dict.items():
        numbers = [tup[0] for tup in v['synergy_number_info']]
        synergy_to_numbers[k] = numbers

    return(synergy_info_dict, synergy_to_numbers)

# --------------------------------------------------------- drawing the graph out ----------------------------------------------------------

def make_graph(G, nodes_to_highlight, nodes, edge_col, col_map):
    plt.figure(figsize=(20, 15))
    
    pos = nx.kamada_kawai_layout(G)

    edge_widths = [5 if (n1 in nodes_to_highlight or n2 in nodes_to_highlight) else 1 for (n1, n2) in G.edges()]
    es = nx.draw_networkx_edges(G, pos=pos, edge_color=edge_col, width=edge_widths)

    edge_labels = nx.get_edge_attributes(G,'label')

    # els = nx.draw_networkx_edge_labels(G, pos = pos, edge_labels=edge_labels)
    ns = nx.draw_networkx_nodes(G, pos=pos, node_color='white', node_size=1) # want tiny nodes so can't see them
    nodes_not_highlight = set(nodes).difference(set(nodes_to_highlight))
    ls = nx.draw_networkx_labels(G, pos = pos, labels = {n:n for n in nodes_not_highlight}, font_size=16) # not to highlight, small
    ls = nx.draw_networkx_labels(G, pos = pos, labels = {n:n for n in nodes_to_highlight}, font_size=26) # to highlight, big

    # Generate legend
    from matplotlib.lines import Line2D
    create_lines = lambda clr, **kwargs: Line2D([0, 1], [0, 1], color=clr, **kwargs)
    lines = [create_lines(clr, lw=5) for clr in col_map.values()]
    labels = [i for i in col_map.keys()]
    plt.legend(lines, labels)
    
# experimenting with positions (and weights):
# fixed_positions = {'Ahri':(0,0)}#dict with two of the positions set
# fixed_nodes = fixed_positions.keys()
# pos = nx.spring_layout(G,pos=fixed_positions, fixed = fixed_nodes)
# pos = nx.fruchterman_reingold_layout(G, weight = 'sg')
# pos = nx.kamada_kawai_layout(G, weight='sg')


# --------------------------------------------------------- helper functions for finding team additions ------------------------------------

# function to work out what synergies you have
def get_synergies(team, synergies_df):
    synergy_count = {}
    for n in team:
        ss = synergies_df[n][0]
        for s in ss:
            if s in synergy_count.keys():
                synergy_count[s] = synergy_count[s] + 1
            else:
                synergy_count[s] = 1
    return(synergy_count)



# returns the difference in synergies between 2 teams (team2 - team1)
def get_synergy_diff(team1, team2, synergies_df):
    diff_dict = {}
    team1_s = get_synergies(team1, synergies_df)
    team2_s = get_synergies(team2, synergies_df)
    for key in list(team1_s.keys()):
        if key in team2_s:
            diff = team2_s[key] - team1_s[key]
            if diff != 0:
                diff_dict[key] = diff
        else:
            diff_dict[key] = -team1_s[key]
    
    for key in list(team2_s.keys()):
        if key not in team1_s: # we already dealt with this case
            diff_dict[key] = team2_s[key]
    
    return(diff_dict)
    

def get_all_combinations(num_to_add, team, nodes):
    to_pick_from = set(nodes) - set(team)
    all_combinations = list(itertools.combinations(to_pick_from, num_to_add))
    return(all_combinations)


# --------------------------------------------------------- finding team additions --------------------------------------------------------

# brutefroce approach - adds all possible combinations of num_to_add new team members and returns the combinations that increase the synergies the most
def more_synergies_please(team, num_to_add, nodes, synergies_df):
    good_additions = []
    count_added = []
    all_combs = get_all_combinations(num_to_add, team, nodes)
    for comb in all_combs:
#         print(team + list(comb))
        synergy_diff = get_synergy_diff(team, team + list(comb), synergies_df)
        # how do we add to our current synergies?
        curr_synergies = get_synergies(team, synergies_df)
#         print(curr_synergies)
        added_synergies = {k:v for k, v in synergy_diff.items() if k in curr_synergies.keys()}
        if added_synergies != {}:
            good_additions.append((comb, added_synergies))
            count = sum([v for v in added_synergies.values()])
            count_added.append(count)
#         print(synergy_diff)
    # order so best choices at the top
    args = np.argsort(count_added)
    args = np.flip(args)
#     print(args)
    sorted_good_additions = [good_additions[i] for i in args]
    return(sorted_good_additions)


# uses result from more_synergies please, and picks out rows where the synergies get to the next level of effect, found in synergies_to_numbers
def get_additions_that_level_synergy_up(good_additions, team, synergies_df, synergy_to_numbers):
    num_level_ups = []
    curr_synergies = get_synergies(team, synergies_df)
    for i in range(len(good_additions)):
        row = good_additions[i]
        num_level_up = 0
        for k, v in row[1].items(): 
            # does it make a synergy number?
            new_num = curr_synergies[k] + v
            if new_num in synergy_to_numbers[k]:
                num_level_up = num_level_up + 1
        num_level_ups.append(num_level_up)
    # now remove all 0s
    num_level_ups = np.array(num_level_ups).astype(int)
    good_additions = np.array(good_additions)[num_level_ups != 0]
    num_level_ups = num_level_ups[num_level_ups != 0]
    
    order = np.argsort(num_level_ups)
    good_additions = good_additions[order]
    good_additions = list(good_additions)
    good_additions.reverse()
    num_level_ups = num_level_ups[order]
    num_level_ups = np.flip(num_level_ups)
    
    return(list(good_additions), list(num_level_ups))




# --------------------------------------------------------- helper functions for finding team replacements --------------------------------

def get_num_counted_synergies(synergies, synergy_to_numbers):
    num_level_up = 0
    for k, v in synergies.items(): 
        # does it make a synergy number?
        new_num = synergies[k]
        if new_num >= synergy_to_numbers[k][0]: # if its bigger than the smallest one
            syn_nums = np.array(synergy_to_numbers[k])
            num_counted = syn_nums[syn_nums <= new_num][-1] # take all items where new_num is bigger or equal and take the largest one (the last one - it's in order)
            num_level_up = num_level_up + num_counted
    return(num_level_up)


def get_num_level_ups(synergies, synergy_to_numbers):
    num_level_up = 0
    for k, v in synergies.items(): 
        new_num = synergies[k]
        if new_num in synergy_to_numbers[k]:
            num_level_up = num_level_up + 1
    return(num_level_up)


# --------------------------------------------------------- finding team replacements ----------------------------------------------------

def replace_characters_please(team, num_to_replace, synergies_df, synergy_to_numbers, nodes, only_include_level_ups = False):
    curr_synergies = get_synergies(team, synergies_df)
    good_replacements = []
    if only_include_level_ups:
        # deal with calcing num_synergies only including level ups
        num_synergies = get_num_counted_synergies(curr_synergies, synergy_to_numbers)
    else:
        num_synergies = sum([v for k, v in curr_synergies.items()])
    combinations = list(itertools.combinations(team, num_to_replace)) # the combinations of characters to remove
    for comb in combinations:
        to_keep = set(team) - set(comb)
        # remove comb from team, replace with other characters and see if there are more synergies
        good_additions = more_synergies_please(list(set(team) - set(comb)), num_to_replace, nodes, synergies_df)
        if only_include_level_ups:
            good_additions, num_level_ups = get_additions_that_level_synergy_up(good_additions, list(set(team) - set(comb)), synergies_df, synergy_to_numbers)

        # now only include if more synergies than in team
        new_synergies = [get_synergies(list(to_keep) + list(poss_replacement[0]), synergies_df) for poss_replacement in good_additions]
        if only_include_level_ups:
            new_synergy_numbers = [get_num_counted_synergies(new) for new in new_synergies]
            num_level_ups = [get_num_level_ups(new, synergy_to_numbers) for new in new_synergies]
        else:
            new_synergy_numbers = [sum(new.values()) for new in new_synergies]
        improved_replacements = np.array(good_additions)[np.array(new_synergy_numbers) > num_synergies]
        new_synergy_numbers = np.array(new_synergy_numbers)[np.array(new_synergy_numbers) > num_synergies]
        
        if only_include_level_ups:
            to_add = [{'replacement': improved_replacements[i], 'to_remove': comb, 'num_new_synergies': new_synergy_numbers[i], 'num_level_ups': num_level_ups[i]} for i in range(len(improved_replacements))]
        else:
            to_add = [{'replacement': improved_replacements[i], 'to_remove': comb, 'num_new_synergies': new_synergy_numbers[i]} for i in range(len(improved_replacements))]
        good_replacements.extend(to_add)
        
    # now to sort it
    if only_include_level_ups:
        good_replacements = sorted(good_replacements, key=lambda k: k['num_level_ups']) 
    else:
        good_replacements = sorted(good_replacements, key=lambda k: k['num_new_synergies']) 
    good_replacements.reverse()
    return(good_replacements)



# --------------------------------------------------------- suggestion functions aimed at the user -----------------------------------------


def suggest_team_additions_for_user(team, synergies_df, synergy_to_numbers, nodes, only_include_level_ups = False):
    pd.options.display.max_rows = 30
    dfs = []
    for num_to_add in [1, 2]: # adding up to 2 more I think
        sorted_good_additions = more_synergies_please(team, num_to_add, nodes, synergies_df)
        if only_include_level_ups:
            sorted_good_additions, num_level_ups = get_additions_that_level_synergy_up(sorted_good_additions, team, synergies_df, synergy_to_numbers)
#         print(sorted_good_additions)
        # make nice dataframe
        df = pd.DataFrame([', '.join(list(i[0])) for i in sorted_good_additions], columns=['Possible addition'])
#         print(df)
        extra_synergies = [i[1] for i in sorted_good_additions]
#         print(extra_synergies)
        extra_synergies = [', '.join([k + ' (' + str(v) + ')' for k, v in e.items()]) for e in extra_synergies]
        df['extra synergies'] = extra_synergies
        dfs.append(df)
    return(dfs)

def suggest_team_replacements_for_user(team, synergies_df, synergy_to_numbers, nodes, only_include_level_ups = False):
    dfs = []
    for num_to_replace in [1, 2]: # replacing up to 2 more I think
        sorted_replacements = replace_characters_please(team, num_to_replace, synergies_df, synergy_to_numbers, nodes, only_include_level_ups)
        # make nice dataframe
        df = pd.DataFrame([', '.join(list(i['replacement'][0])) for i in sorted_replacements], columns=['Possible addition'])
        
        to_remove = [', '.join(i['to_remove']) for i in sorted_replacements]
        df['To remove'] = to_remove
        
        extra_synergies = [i['replacement'][1] for i in sorted_replacements]
#         print(extra_synergies)
        extra_synergies = [', '.join([k + ' (' + str(v) + ')' for k, v in e.items()]) for e in extra_synergies]
        df['extra synergies'] = extra_synergies
        
        dfs.append(df)
    return(dfs)


# --------------------------------------------------------- making the team by clicking on characters --------------------------------------


from ipywidgets import Label, HTML, HBox, Image, VBox, Box, HBox
from ipyevents import Event 
from IPython.display import display

# this depends on what is in character_pic!
def get_char_name(col, row, nodes):
    chars = sorted(nodes)
    chars = chars + ['', ''] # as there are 58 characters (NOTE: hard coded!), we add 2 empty ones to make it work in an array
    chars = np.reshape(np.array(chars), (6, 10))
#     print(chars[row, col])
    return(chars[row, col])


def prep_team_image():
    with open('character_pic.png', 'rb') as f:
        value = f.read()
    
    image = Image(value=value, format='png')

    # The layout bits below make sure the image display looks the same in lab and classic notebook 
    image.layout.max_width = '100%'
    image.layout.height = 'auto'

    im_events = Event()
    im_events.source = image
    im_events.watched_events = ['click']

    no_drag = Event(source=image, watched_events=['dragstart'], prevent_default_action = True)

    vbox = VBox()
    vbox.layout.width = "100%"

    images = HBox()
    i1 = Box()
    i1.layout.width = '1000px'
    # i1.layout.border = '2px solid black'
    i1.children = [image]

    images.children = [i1]

    coordinates = HTML('<h3>Click a team member to add them to your team!</h3>')

    vbox.children = [images, coordinates]
    
    return(im_events, vbox, coordinates)




















