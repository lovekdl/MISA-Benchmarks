import re
sentence ="AsBsCsDsEs\n"

sentence_ = sentence.strip()
# sentence_ = ''.join(reversed(sentence_))
pred_answers = re.findall(r'A|B|C|D|E', sentence_)
if pred_answers:
    print(pred_answers[0])
else:
    print("die")