import streamlit as st
import sys
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入transformers库
logger.info(f"Python version: {sys.version}")
try:
    import transformers
    logger.info(f"transformers version: {transformers.__version__}")
    # 只导入实际使用的组件，移除未使用的AutoModelForCausalLM
    from transformers import AutoTokenizer, AutoModel
    logger.info("Successfully imported AutoTokenizer and AutoModel")
except Exception as e:
    logger.error(f"Error importing transformers: {e}")
    raise

import torch
import numpy as np
from sklearn.preprocessing import normalize

# 添加国际化支持
import json
import os

# 加载翻译文件
def load_translations():
    translations = {}
    locales_dir = "locales"
    for lang_file in os.listdir(locales_dir):
        if lang_file.endswith(".json"):
            lang = lang_file[:-5]  # 移除.json后缀
            with open(os.path.join(locales_dir, lang_file), "r", encoding="utf-8") as f:
                translations[lang] = json.load(f)
    return translations

# 加载翻译
translations = load_translations()

# 设置页面标题和布局
st.set_page_config(
    page_title="Transformer ChatGPT 可视化演示",
    page_icon="🤖",
    layout="wide"
)

# 添加自定义CSS来隐藏右上角的部署和分享按钮
st.markdown("""<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)

# 侧边栏：参数设置
with st.sidebar:
    # 语言选择器
    lang = st.selectbox(
        "选择语言 / Select Language",
        options=["zh", "en"],
        index=0,
        format_func=lambda x: "中文" if x == "zh" else "English"
    )
    st.markdown("---")

# 获取翻译函数
def t(key):
    return translations[lang].get(key, key)

# 加载预训练模型和分词器（仅在首次运行时加载）
@st.cache_resource
def load_models():
    # 使用支持多语言的模型，解决英文单词被标记为UNK的问题
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    embedding_model = model
    model.eval()
    embedding_model.eval()
    return tokenizer, model, embedding_model

tokenizer, model, embedding_model = load_models()

# 计算两个向量的余弦相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 主标题
st.title(t("title"))

# 侧边栏：参数设置
with st.sidebar:
    st.header(t("sidebar_header"))
    
    # 用户输入
    user_input = st.text_input(t("user_input"), "你好，我想了解Transformer", max_chars=200, help=t("max_chars_help"))
    
    # 生成参数
    temperature = st.slider(t("temperature"), 0.1, 2.0, 0.7, 0.1, help=t("temperature_help"))
    top_p = st.slider(t("top_p"), 0.1, 1.0, 0.95, 0.05, help=t("top_p_help"))
    top_k = st.slider(t("top_k"), 10, 200, 50, 10, help=t("top_k_help"))
    max_new_tokens = st.slider(t("max_new_tokens"), 50, 500, 200, 50, help=t("max_new_tokens_help"))
    
    # 处理步骤控制
    st.header(t("processing_steps"))
    
    # 基础处理步骤
    step_1 = st.checkbox(t("tokenization"), True)
    step_2 = st.checkbox(t("encoding"), True)
    step_3 = st.checkbox(t("vectorization"), True)
    step_4 = st.checkbox(t("normalization"), True)
    step_5 = st.checkbox(t("correlation"), True)
    step_6 = st.checkbox(t("generation"), True)
    
    # 高级功能分组
    st.markdown("---")
    st.subheader(t("advanced_features"))
    advanced_features = st.checkbox(t("enable_advanced"), False)
    
    if advanced_features:
        # MCP (Model Context Processing) 演示
        step_7 = st.checkbox(t("mcp"), True)
        # Skill 演示
        step_8 = st.checkbox(t("skill"), True)
        # RAG (检索增强生成) 演示
        step_9 = st.checkbox(t("rag"), True)
    else:
        step_7 = False
        step_8 = False
        step_9 = False
    
    # 开始处理按钮
    process_button = st.button(t("process_button"), type="primary")

# 主内容区域
if process_button:
    # 对话历史（在实际应用中可以保存）
    conversation_history = f"用户: {user_input}\nAI: "
    
    # 初始化结果存储
    results = {}
    
    # 添加进度条
    progress_bar = st.progress(0, text=t("processing"))
    total_steps = sum([step_1, step_2, step_3, step_4, step_5, step_6, step_7, step_8, step_9])
    current_step = 0
    
    # 1. 分词过程
    if step_1:
        with st.expander(f"🔤 {t('tokenization')}", expanded=True):
            st.info(t("tokenization_tip"))
            tokens = tokenizer.tokenize(user_input)
            results['tokens'] = tokens
            st.write(f"{t('original_input')} {user_input}")
            st.write(f"{t('tokenization_result')} {tokens}")
            
            # 可视化分词
            st.write(f"{t('tokenization_visualization')}")
            for i, token in enumerate(tokens):
                st.code(f"Token {i+1}: '{token}'")
        current_step += 1
        progress_bar.progress(current_step / total_steps, text=f"{t('tokenization_complete')} ({current_step}/{total_steps})")
    
    # 2. 编码过程
    if step_2:
        with st.expander(f"🔢 {t('encoding')}", expanded=True):
            st.info(t("encoding_tip"))
            token_ids = tokenizer.encode(user_input, add_special_tokens=False)
            results['token_ids'] = token_ids
            st.write(f"{t('token_ids')} {token_ids}")
            
            # 分词与ID对应关系
            if step_1:
                st.write(f"{t('token_id_mapping')}")
                for token, token_id in zip(results['tokens'], token_ids):
                    st.code(f"'{token}' → {token_id}")
        current_step += 1
        progress_bar.progress(current_step / total_steps, text=f"{t('encoding_complete')} ({current_step}/{total_steps})")
    
    # 3. 向量化过程
    if step_3:
        with st.expander(f"📊 {t('vectorization')}", expanded=True):
            st.info(t("vectorization_tip"))
            
            # 将token IDs转换为PyTorch张量
            input_ids = torch.tensor([results['token_ids']])
            
            # 显示token到向量的转换方法
            st.subheader(t("token_to_vector"))
            st.write(t("word_embedding"))
            st.write(t("positional_embedding"))
            st.write(t("layer_norm"))
            st.write(t("multi_head_attention"))
            st.write(t("feed_forward"))
            
            # 可视化点积计算过程
            st.subheader(t("dot_product_visualization"))
            st.write(t("dot_product_desc"))
            
            # 简单的点积示例
            vec_a = np.array([0.5, 0.7, 0.2])
            vec_b = np.array([0.3, 0.6, 0.9])
            dot_product = np.dot(vec_a, vec_b)
            
            col1, col2, col3 = st.columns(3)
            col1.write(f"{t('vector_a')}")
            col1.write(vec_a)
            col2.write(f"{t('vector_b')}")
            col2.write(vec_b)
            col3.write(f"{t('dot_product_result')}")
            col3.write(f"{dot_product:.4f}")
            
            st.write(t("dot_product_calc"))
            
            # Softmax过程描述和可视化
            st.subheader(t("softmax_activation"))
            st.info(t("softmax_tip"))
            
            # 简单的softmax示例
            logits = np.array([2.0, 1.0, 0.1])
            exp_logits = np.exp(logits)
            softmax_probs = exp_logits / np.sum(exp_logits)
            
            st.write(f"{t('input_logits')} {logits}")
            st.write(f"{t('exponential_transform')} {exp_logits}")
            st.write(f"{t('sum_result')} {np.sum(exp_logits):.4f}")
            st.write(f"{t('softmax_probs')} {softmax_probs}")
            st.write(f"{t('probability_sum')} {np.sum(softmax_probs):.4f}")
            
            # 可视化softmax曲线
            st.bar_chart({
                'Logits': logits,
                'Softmax Probabilities': softmax_probs
            })
            
            # 获取模型的隐藏状态和注意力权重
            with torch.no_grad():
                # 设置output_attentions=True以获取注意力权重
                outputs = embedding_model(input_ids, output_attentions=True)
                last_hidden_state = outputs.last_hidden_state
                attentions = outputs.attentions
                sentence_vector = last_hidden_state.mean(dim=1).squeeze().numpy()
            
            results['sentence_vector'] = sentence_vector
            results['attentions'] = attentions
            
            # 突出注意力关系
            st.subheader(t("attention_visualization"))
            st.info(t("attention_tip"))
            
            if attentions is not None and len(tokens) > 0:
                # 获取最后一层的注意力权重
                last_layer_attention = attentions[-1].squeeze(0).numpy()  # 形状: (num_heads, seq_len, seq_len)
                num_heads = last_layer_attention.shape[0]
                
                st.write(f"{t('attention_heads')} {num_heads}")
                
                # 选择一个注意力头进行可视化（这里选择第0个）
                attention_head = 0
                attention_matrix = last_layer_attention[attention_head]
                
                # 确保tokens数量与注意力矩阵维度一致
                seq_len = attention_matrix.shape[0]
                if len(tokens) < seq_len:
                    # 如果tokens数量不足，用空字符串填充
                    display_tokens = tokens + ["[PAD]"] * (seq_len - len(tokens))
                else:
                    display_tokens = tokens[:seq_len]
                
                # 计算显示的注意力头编号（从1开始）
                head_num = attention_head + 1
                
                # 显示注意力热力图
                st.write(t("attention_matrix").format(head_num=head_num))
                st.write(t("attention_cell_desc"))
                
                # 创建注意力权重DataFrame用于热力图
                import pandas as pd
                attention_df = pd.DataFrame(attention_matrix, index=display_tokens, columns=display_tokens)
                st.dataframe(attention_df.style.background_gradient(cmap='viridis', axis=None))
                
                # 显示注意力权重的最大值
                max_attention = attention_matrix.max()
                max_pos = np.unravel_index(attention_matrix.argmax(), attention_matrix.shape)
                # 计算显示的token编号（从1开始）
                token_from = max_pos[0] + 1
                token_to = max_pos[1] + 1
                st.write(t("max_attention").format(max_attn=max_attention, token_from=token_from, token_to=token_to))
            
            st.subheader(t("final_vector"))
            st.write(f"{t('vector_dimension')} {sentence_vector.shape}")
            st.write(f"{t('vector_first_20')} {sentence_vector[:20]}")
            
            # 向量统计信息
            st.write(f"{t('vector_stats')}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(t("min_value"), f"{np.min(sentence_vector):.4f}")
            col2.metric(t("max_value"), f"{np.max(sentence_vector):.4f}")
            col3.metric(t("mean_value"), f"{np.mean(sentence_vector):.4f}")
            col4.metric(t("std_value"), f"{np.std(sentence_vector):.4f}")
        current_step += 1
        progress_bar.progress(current_step / total_steps, text=f"{t('vectorization_complete')} ({current_step}/{total_steps})")
    
    # 4. 归一化过程
    if step_4:
        with st.expander(f"📏 {t('normalization')}", expanded=True):
            st.info(t("normalization_tip"))
            sentence_vector = results['sentence_vector']
            normalized_vector = normalize([sentence_vector], norm='l2')[0]
            results['normalized_vector'] = normalized_vector
            
            st.write(f"{t('norm_before')} {np.linalg.norm(sentence_vector):.6f}")
            st.write(f"{t('norm_after')} {np.linalg.norm(normalized_vector):.6f}")
            st.write(f"{t('normalized_vector')} {normalized_vector[:20]}")
            
            # 可视化归一化前后的向量变化
            st.write(f"{t('norm_comparison')}")
            col1, col2 = st.columns(2)
            col1.write(f"{t('before_normalization')}")
            col1.line_chart(sentence_vector[:50])
            col2.write(f"{t('after_normalization')}")
            col2.line_chart(normalized_vector[:50])
        current_step += 1
        progress_bar.progress(current_step / total_steps, text=f"{t('normalization_complete')} ({current_step}/{total_steps})")
    
    # 5. 相关性计算（与预设的示例向量）
    if step_5:
        with st.expander(f"🔗 {t('correlation')}", expanded=True):
            st.info(t("correlation_tip"))
            # 预设一些示例向量用于相关性计算
            example_vectors = {
                "问候": np.random.randn(768),
                "技术": np.random.randn(768),
                "娱乐": np.random.randn(768),
                "教育": np.random.randn(768),
                "健康": np.random.randn(768)
            }
            
            # 归一化示例向量
            for key in example_vectors:
                example_vectors[key] = normalize([example_vectors[key]], norm='l2')[0]
            
            # 计算相关性
            similarities = {}
            for key, vec in example_vectors.items():
                sim = cosine_similarity(results['normalized_vector'], vec)
                similarities[key] = sim
            
            st.write(f"{t('correlation_with_categories')}")
            for key, sim in similarities.items():
                st.progress(float((sim + 1) / 2), text=f"{key}: {sim:.4f}")
            
            # 显示最高相关性
            most_relevant = max(similarities, key=similarities.get)
            similarity_value = similarities[most_relevant]
            st.write(t("most_relevant").format(most_relevant=most_relevant, similarity_value=similarity_value))
        current_step += 1
        progress_bar.progress(current_step / total_steps, text=f"{t('correlation_complete')} ({current_step}/{total_steps})")
    
    # 6. 文本分类（BERT是编码器模型，用于分类而不是生成）
    if step_6:
        with st.expander(f"🏷️ {t('generation')}", expanded=True):
            st.info(t("classification_tip"))
            st.write(f"{t('bert_analysis')}")
            st.info(t("bert_info"))
            
            # 添加Transformer架构类型讲解
            st.subheader("Transformer Architecture Types")
            st.markdown("""
            Transformer模型主要分为三种架构类型，每种架构有不同的应用场景：
            
            ### 1. Encoder-only Architecture
            **代表模型**: BERT, RoBERTa, ALBERT, DistilBERT
            **特点**:
            - 仅包含Transformer编码器部分
            - 双向注意力，能同时看到上下文信息
            - 适合理解类任务
            **应用场景**:
            - 文本分类
            - 命名实体识别
            - 情感分析
            - 文本相似度计算
            - 信息检索
            
            ### 2. Decoder-only Architecture
            **代表模型**: GPT系列, Llama, Mistral, Gemma
            **特点**:
            - 仅包含Transformer解码器部分
            - 单向注意力，只能看到之前的信息
            - 适合生成类任务
            **应用场景**:
            - 文本生成
            - 对话系统
            - 故事创作
            - 代码生成
            - 自动写作
            
            ### 3. Encoder-Decoder Architecture
            **代表模型**: T5, BART, mT5, Pegasus
            **特点**:
            - 包含完整的编码器和解码器
            - 编码器处理输入，解码器生成输出
            - 适合序列到序列任务
            **应用场景**:
            - 机器翻译
            - 文本摘要
            - 问答系统
            - 文本改写
            - 语音识别
            
            ### Architecture Comparison
            | Architecture | Key Features | Typical Tasks | Representative Models |
            |--------------|--------------|---------------|------------------------|
            | Encoder-only | Bidirectional attention | Understanding tasks | BERT, RoBERTa |
            | Decoder-only | Unidirectional attention | Generation tasks | GPT, Llama |
            | Encoder-Decoder | Both encoder and decoder | Sequence-to-sequence tasks | T5, BART |
            
            This demo uses BERT, an encoder-only model, which is why it excels at understanding and classification tasks but doesn't support text generation like GPT models.
            """)
            
            st.write(f"{t('vector_applications')}")
            st.write(t("similarity_calc"))
            st.write(t("text_classification"))
            st.write(t("information_retrieval"))
            st.write(t("clustering"))
            st.write(t("recommendation"))
        current_step += 1
        progress_bar.progress(current_step / total_steps, text=f"{t('generation_complete')} ({current_step}/{total_steps})")
    
    # 7. MCP (Model Context Processing) 演示
    if step_7:
        with st.expander(f"🧠 {t('mcp')}", expanded=True):
            st.info(t("mcp_tip"))
            
            st.subheader(t("mcp_process"))
            
            # 展示上下文处理的不同阶段
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(t("context_acquisition"))
                st.write(t("context_acquisition_desc"))
            
            with col2:
                st.markdown(t("context_optimization"))
                st.write(t("context_optimization_desc"))
            
            with col3:
                st.markdown(t("context_injection"))
                st.write(t("context_injection_desc"))
            
            # 可视化示例
            st.subheader(t("mcp_example"))
            
            # 原始上下文
            original_context = "用户: 你好\nAI: 你好！有什么可以帮助你的？\n用户: 我想了解Transformer\nAI: Transformer是一种基于自注意力机制的深度学习模型...\n用户: 那它和RNN有什么区别？"
            
            # 优化后的上下文
            optimized_context = "任务: 解释Transformer和RNN的区别\n相关历史: 用户询问了Transformer的基本信息\n当前查询: 它和RNN有什么区别？"
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"{t('original_context')}")
                st.code(original_context, language="text")
            with col2:
                st.markdown(f"{t('optimized_context')}")
                st.code(optimized_context, language="text")
            
            st.write(f"{t('mcp_advantages')}")
            st.write(t("mcp_advantage1"))
            st.write(t("mcp_advantage2"))
            st.write(t("mcp_advantage3"))
            st.write(t("mcp_advantage4"))
        current_step += 1
        progress_bar.progress(current_step / total_steps, text=f"{t('mcp_complete')} ({current_step}/{total_steps})")
    
    # 8. Skill (技能调用) 演示
    if step_8:
        with st.expander(f"🛠️ {t('skill')}", expanded=True):
            st.info(t("skill_tip"))
            
            st.subheader(t("skill_process"))
            
            # 技能调用的基本步骤
            skill_steps = [
                {"name": t("intent_recognition"), "desc": t("intent_recognition_desc")},
                {"name": t("skill_selection"), "desc": t("skill_selection_desc")},
                {"name": t("parameter_extraction"), "desc": t("parameter_extraction_desc")},
                {"name": t("skill_execution"), "desc": t("skill_execution_desc")},
                {"name": t("result_integration"), "desc": t("result_integration_desc")}
            ]
            
            for step in skill_steps:
                with st.container(border=True):
                    st.markdown(f"{step['name']}")
                    st.write(step['desc'])
            
            # 技能示例
            st.subheader(t("skill_examples"))
            
            # 示例技能列表
            skills = {
                "计算器": "执行数学计算",
                "天气查询": "获取指定城市的天气信息",
                "日期时间": "获取当前日期和时间",
                "翻译": "将文本翻译为指定语言",
                "搜索引擎": "搜索互联网获取相关信息"
            }
            
            st.write(f"{t('available_skills')}")
            for skill_name, skill_desc in skills.items():
                st.code(f"{skill_name}: {skill_desc}")
            
            # 演示技能调用过程
            st.subheader(t("skill_demonstration"))
            
            # 模拟技能调用
            user_request = "100的平方根是多少？"
            skill_selected = "计算器"
            parameters = {"expression": "sqrt(100)"}
            skill_result = 10.0
            final_response = f"100的平方根是{skill_result}。"
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"{t('user_request')}")
                st.code(user_request)
                st.markdown(f"{t('selected_skill')}")
                st.code(skill_selected)
                st.markdown(f"{t('skill_parameters')}")
                st.code(parameters)
            with col2:
                st.markdown(f"{t('skill_result')}")
                st.code(str(skill_result))
                st.markdown(f"{t('final_response')}")
                st.code(final_response)
        current_step += 1
        progress_bar.progress(current_step / total_steps, text=f"{t('skill_complete')} ({current_step}/{total_steps})")
    
    # 9. RAG (检索增强生成) 演示
    if step_9:
        with st.expander(f"🔍 {t('rag')}", expanded=True):
            st.info(t("rag_tip"))
            
            st.subheader(t("rag_process"))
            
            # RAG的核心组件和流程
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(t("rag_components"))
                st.write(t("retriever"))
                st.write(t("generator"))
                st.write(t("document_library"))
                st.write(t("indexer"))
            
            with col2:
                st.markdown(t("rag_workflow"))
                st.write(t("rag_workflow1"))
                st.write(t("rag_workflow2"))
                st.write(t("rag_workflow3"))
                st.write(t("rag_workflow4"))
                st.write(t("rag_workflow5"))
            
            # 可视化RAG流程
            st.subheader(t("rag_visualization"))
            
            # 创建一个简单的RAG流程图
            st.markdown('''```
用户查询 → 向量转换 → 文档检索 → 结果整合 → 模型生成 → 最终响应
          ↑           ↑
          |           |
   嵌入模型       文档库
            ```''')
            
            # 示例演示
            st.subheader(t("rag_example"))
            
            # 模拟RAG过程
            rag_query = "Transformer的注意力机制是如何工作的？"
            
            # 模拟检索结果
            retrieved_docs = [
                {"title": "Transformer注意力机制详解", "content": "Transformer使用自注意力机制来计算输入序列中每个位置与其他位置的关联程度..."},
                {"title": "注意力机制在NLP中的应用", "content": "注意力机制允许模型在处理每个位置时关注输入序列中的相关位置..."},
                {"title": "Transformer架构解析", "content": "多头注意力是Transformer的核心组件，它允许模型从不同角度关注输入信息..."}
            ]
            
            st.write(f"{t('rag_query')}")
            st.code(rag_query)
            
            st.write(f"{t('retrieved_documents')}")
            for doc in retrieved_docs:
                with st.container(border=True):
                    st.markdown(f"### {doc['title']}")
                    st.write(doc['content'][:150] + "...")
            
            st.write(f"{t('rag_advantages')}")
            st.write(t("rag_advantage1"))
            st.write(t("rag_advantage2"))
            st.write(t("rag_advantage3"))
            st.write(t("rag_advantage4"))
        current_step += 1
        progress_bar.progress(current_step / total_steps, text=f"{t('rag_complete')} ({current_step}/{total_steps})")
    
    # 处理完成
    progress_bar.progress(1.0, text=t("processing_completed"))
    st.success(t("all_steps_completed"))

# 页脚
st.markdown("---")
st.markdown("**Transformer Visual Demo - Based on BERT-base-multilingual-cased Model**")
