from langchain_community.retrievers import WikipediaRetriever,wikipedia

wikipedia.USER_AGENT = "MyLangChainApp/1.0 (ayushrawal2020@gamil.com)"

retriver=WikipediaRetriever(top_k_results=2,lang="en")

query=("what is Cricket")

try:
    docs=retriver.invoke(query)
except Exception as e:
    print(f"Retrieval failed: {e}")
    docs = []

for i ,doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")  