from llm_sdk import Small_LLM_Model
import numpy

model = Small_LLM_Model()
text = "The capital of France is"

ids = model.encode(text)

int_list: list = ids[0].tolist()

for _ in range(20):
    logits = model.get_logits_from_input_ids(int_list)
    id_token = numpy.argmax(logits)
    int_list.append(int(id_token))

result = model.decode(int_list)
print(result)
