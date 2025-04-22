from transformers import T5ForConditionalGeneration, AutoTokenizer
import os
import time
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# Load model and tokenizer
model_name = "Salesforce/codet5p-220m"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = model.to('cuda')


torch.manual_seed(0)
torch. cuda.empty_cache()
model.eval()


batch_size = 32
input_ids = torch.full((batch_size , 1024), fill_value=17)
decoder_input_ids = torch.full((batch_size, 256), fill_value=20)
input_ids= input_ids.to('cuda')
decoder_input_ids = decoder_input_ids.to('cuda')
start_time = time.time()
with torch.no_grad():
    output = model.generate(input_ids=input_ids, decoder_input_ids=decoder_input_ids, do_sample=True, max_new_tokens=20,use_cache=True)
    torch.cuda.synchronize()
end_time = time.time()
total_time = end_time - start_time

time_per_sequence = total_time / (batch_size)

print(f"total time: {total_time}")
print(f"time per sequence: {time_per_sequence}")