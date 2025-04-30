from policy_code.alphazero import AlphaZero

class BasePolicy:
    def __init__(self):
        pass

    

    def get_action(self, state):
        raise NotImplementedError
    
class AlphaZero(BasePolicy):
    def __init__(self, ):
        super().__init__()
        

    def get_action(self, state): 
        pass
