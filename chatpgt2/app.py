from flask import Flask, request, render_template, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the pretrained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("place your chat output here")
tokenizer = GPT2Tokenizer.from_pretrained("place your chat output here")

# Function to generate AI response
def generate(prompt):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    eos_id = tokenizer.eos_token_id

    sample_outputs = model.generate(
        input_ids,
        max_length=len(input_ids[0]) + 50,
        min_length=len(input_ids[0]) + 30,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        temperature=0.95,
        num_beams=1,
        eos_token_id=eos_id,
        num_return_sequences=1
    )

    for sample_output in sample_outputs:
        sample_output = sample_output[len(input_ids[0]):]
        return tokenizer.decode(sample_output, skip_special_tokens=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']
    prompt = '<start>Hello.<start_chat>' + '<Human>' + user_input + '<AI>'
    generated = generate(prompt)
    if '<Human>' in generated:
        generated = generated[:generated.find('<Human>')]
    return jsonify({'response': generated})

if __name__ == '__main__':
    app.run(debug=True)
