from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file if present
# Initialize client (it will use your OPENAI_API_KEY from environment variables)
client = OpenAI()

def test_openai():
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # you can change to "gpt-4.1" or others
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you respond to confirm the API is working?"}
            ],
            max_tokens=50
        )

        print("✅ API Response:")
        print(response.choices[0].message.content)

    except Exception as e:
        print("❌ Error calling OpenAI API:", e)

if __name__ == "__main__":
    test_openai()
