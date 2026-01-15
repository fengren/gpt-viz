from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import numpy as np
from sklearn.preprocessing import normalize

# 加载预训练模型和分词器
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
# 用于生成嵌入向量的模型
embedding_model = AutoModel.from_pretrained("gpt2")

# 设置模型为评估模式
model.eval()
embedding_model.eval()

print("欢迎使用Transformer ChatGPT演示！输入'退出'结束对话。")
print("本演示将展示用户输入的分词、向量化、相关性计算和归一化过程。")

# 对话历史
conversation_history = ""
# 对话历史向量列表
history_vectors = []

# 计算两个向量的余弦相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

while True:
    # 获取用户输入
    user_input = input("用户: ")
    
    if user_input == "退出":
        print("感谢使用，再见！")
        break
    
    print("\n=== 处理过程展示 ===")
    
    # 1. 分词过程
    print("1. 分词结果:")
    tokens = tokenizer.tokenize(user_input)
    print(f"   原始输入: {user_input}")
    print(f"   分词结果: {tokens}")
    
    # 2. 编码过程（转换为token ID）
    print("2. 编码结果（Token IDs）:")
    token_ids = tokenizer.encode(user_input, add_special_tokens=False)
    print(f"   Token IDs: {token_ids}")
    
    # 3. 向量化过程
    print("3. 向量化过程:")
    # 将token IDs转换为PyTorch张量
    input_ids = torch.tensor([token_ids])
    # 获取模型的隐藏状态作为向量表示
    with torch.no_grad():
        outputs = embedding_model(input_ids)
        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state
        # 对所有token的向量进行平均，得到句子向量
        sentence_vector = last_hidden_state.mean(dim=1).squeeze().numpy()
    print(f"   向量维度: {sentence_vector.shape}")
    print(f"   向量前10个元素: {sentence_vector[:10]}")
    
    # 4. 归一化过程
    print("4. 归一化过程:")
    # L2归一化
    normalized_vector = normalize([sentence_vector], norm='l2')[0]
    print(f"   归一化前向量范数: {np.linalg.norm(sentence_vector):.6f}")
    print(f"   归一化后向量范数: {np.linalg.norm(normalized_vector):.6f}")
    print(f"   归一化后向量前10个元素: {normalized_vector[:10]}")
    
    # 5. 相关性计算
    print("5. 相关性计算:")
    if history_vectors:
        # 计算与历史对话的相关性
        similarities = []
        for i, hist_vec in enumerate(history_vectors):
            sim = cosine_similarity(normalized_vector, hist_vec)
            similarities.append(sim)
        print(f"   与历史对话的相似度: {similarities}")
        print(f"   最高相似度: {max(similarities):.6f} (第{np.argmax(similarities)+1}轮对话)")
    else:
        print("   暂无历史对话，无法计算相关性")
    
    # 更新对话历史
    conversation_history += f"用户: {user_input}\nAI: "
    # 保存当前输入的归一化向量到历史
    history_vectors.append(normalized_vector)
    
    # 编码完整的对话历史
    inputs = tokenizer.encode(conversation_history, return_tensors="pt")
    
    # 生成响应
    outputs = model.generate(
        inputs,
        max_length=1000,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 解码响应
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取AI的响应
    ai_response = response.split("AI: ")[-1]
    
    # 更新对话历史
    conversation_history += f"{ai_response}\n"
    
    # 打印响应
    print(f"\nAI: {ai_response}")
    print("\n" + "="*50 + "\n")