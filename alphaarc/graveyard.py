"""def _compute_score_from_logits(self, actions, logits): 
        probabilities = F.softmax(logits, dim=-1) 
        chosen_token_probs = probabilities.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        chosen_token_probs = torch.clamp(chosen_token_probs, min=1e-12)
        log_seq_scores = torch.log(chosen_token_probs).sum(dim=-1)
        normalized_scores = F.softmax(log_seq_scores, dim=0)
        return normalized_scores"""
    

"""def _batch_compute_score(self, actions, logits):
        probabilities = F.softmax(logits, dim=-1)
        
        chosen_token_probs = probabilities.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        
        chosen_token_probs = torch.clamp(chosen_token_probs, min=1e-12)
        
        log_seq_scores = torch.log(chosen_token_probs).sum(dim=-1)
        
        normalized_scores = F.softmax(log_seq_scores, dim=1)
        
        return normalized_scores"""

""" def forward(self, state, actions): 
        
        B, L = state.shape
        B, A, AL = actions.shape

        logits = []
        for i in range(B):
            output_logits = self.model(input_ids=state[i].repeat(A, 1), 
                                decoder_input_ids=actions[i], 
                                use_cache=False).logits
            logits.append(output_logits)

        logits = torch.stack(logits)
        scores = self._batch_compute_score(actions, logits)
        return scores


    def value_forward(self, state):
        return self._batch_compute_values(states=state)
        
        def _compute_values(self, task, state): 
        if state.shape == (1, 0): # TODO: just patched over this for now.
            return torch.tensor([0.0], device=self.device)
        last_hidden_state = self.model.forward(input_ids=task, decoder_input_ids=state, output_hidden_states=True).decoder_hidden_states[-1]
        values = self.value(last_hidden_state)
        values = values.squeeze()
        values = values[-1] # just take the last value prediction
        return values
    
    def _batch_compute_values(self, states): 
        last_hidden_states = self.model.encoder(input_ids=states, use_cache=False, output_hidden_states=True).hidden_states[-1]
        values = self.value(last_hidden_states).squeeze()
        return values[:, -1]
    """