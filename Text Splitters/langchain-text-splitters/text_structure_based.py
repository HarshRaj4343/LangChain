from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. 
Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)



# # Here is the default priority list it follows, from highest to lowest:

# Paragraphs (\n\n) – It first tries to keep entire paragraphs intact.

# Sentences / Lines (\n) – If a paragraph is too large, it splits by line breaks to keep sentences together.

# Words (" ") – If a single sentence or line exceeds the chunk size, it splits by spaces.

# Characters ("") – As a last resort, if a word or continuous string of text is still too long, it splits character by character.