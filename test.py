import torch
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

input_ids = torch.tensor([[101, 6207, 1024, 1037, 103, 16935, 102, 3965, 102, 0], [101, 3207, 1024, 103, 332, 2974, 16935, 102, 3965, 102]])
token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

model = AutoModel.from_pretrained('bert-base-uncased')
outputs = model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
last_hidden_state = outputs['last_hidden_state']
print(outputs['last_hidden_state'].size()) # (2, 10, 768)

bs, length = input_ids.size(0), input_ids.size(1)
# head_tokens = token_type_ids ^ attention_mask
# head_tokens[:, 0] = 0
# token_num = head_tokens.sum(dim=1).unsqueeze(-1)
# head_tokens = head_tokens.unsqueeze(-1).expand(bs, length, last_hidden_state.shape[-1])
# head_embedding = head_tokens * last_hidden_state
# head_embedding = head_embedding.sum(dim=1) / token_num
tokenzier = AutoTokenizer.from_pretrained('bert-base-uncased')
print(tokenzier.pad_token_id)


