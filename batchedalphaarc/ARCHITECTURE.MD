

Changes needed:
    buffer.py
        - Multiple cores need to be able to write to this. This is the same for both the trajectory buffer and the supervised buffer.


    agent.py
        - the entry point for making it multi-core. was going to make the MCTS code the parallelized section, but realize that the mcts interplays with the agent. 
        - agents would take from a queue of tasks (the curriculum to process them)
    
    mcts.py
        - rather than just making a call. It would use like an await functions to make a like maybe call.

    network.py
        - the network thing would likely be its own thread. essentially just waiting till all the agents have submitted their requests.

Tools to use:
    multiprocessing




