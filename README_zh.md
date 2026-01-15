# Transformer ChatGPT 可视化演示

一个基于Transformer模型的ChatGPT最小版本演示，包含完整的处理流程可视化。

## 功能特性

### 基础功能
- 🔤 分词处理：将连续文本分解为有意义的token
- 🔢 编码转换：将token转换为模型可理解的数字ID
- 📊 向量化：将token ID转换为连续的向量表示
- 📏 归一化：将向量缩放到固定范数
- 🔗 相关性计算：衡量文本与其他类别的相似程度
- 🏷️ 文本分类：基于向量表示的文本分类

### 高级功能
- 🧠 MCP (模型上下文处理)：优化模型输入，提升理解能力
- 🛠️ Skill (技能调用)：允许模型使用外部工具和API
- 🔍 RAG (检索增强生成)：结合信息检索和生成式AI

### 可视化特性
- 📊 实时更新的加载进度条
- 👁️ 注意力关系热力图
- 📈 Softmax激活函数可视化
- 🔢 点积计算可视化

## 快速开始

### 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 运行应用

```bash
python -m streamlit run visual_transformer.py
```

然后在浏览器中访问 `http://localhost:8501`

## 部署到 Vercel

### 准备工作

1. 在 GitHub 上创建一个仓库，将项目代码推送上去
2. 在 Vercel 上创建一个新的项目，关联到你的 GitHub 仓库
3. 在 GitHub 仓库的 `Settings > Secrets and variables > Actions` 中添加以下 secrets：
   - `VERCEL_TOKEN`：从 Vercel 账户设置中获取
   - `VERCEL_ORG_ID`：从 Vercel 项目设置中获取
   - `VERCEL_PROJECT_ID`：从 Vercel 项目设置中获取

### 自动部署

当你将代码推送到 `main` 分支时，GitHub Action 会自动触发部署流程，将应用部署到 Vercel 上。

### 手动部署

你也可以使用 Vercel CLI 手动部署：

```bash
npm install -g vercel
vercel login
vercel deploy --prod
```

## 项目结构

```
transformer_demo/
├── visual_transformer.py   # 主应用文件
├── simple_implementation.py # 简单命令行实现
├── requirements.txt        # 项目依赖
├── vercel.json             # Vercel 配置文件
├── .github/workflows/
│   └── deploy.yml          # GitHub Action 工作流
├── README.md               # 英文项目文档
└── README_zh.md            # 中文项目文档
```

## 技术栈

- **框架**: Streamlit
- **模型**: BERT-base-multilingual-cased
- **库**: Transformers, PyTorch, NumPy, Scikit-learn, Matplotlib
- **部署**: Vercel, GitHub Actions

## 使用说明

1. 在侧边栏输入查询内容（最多200个字符）
2. 选择要演示的处理步骤
3. 勾选"启用高级功能"以使用MCP、Skill和RAG功能
4. 点击"开始处理"按钮
5. 展开每个步骤查看详细的处理过程和可视化

## 注意事项

- 首次运行时需要下载预训练模型，可能需要一些时间
- 部分高级功能需要较大的计算资源
- 在Vercel上部署时，可能需要调整模型大小或使用更强大的运行时

## 许可证

MIT
