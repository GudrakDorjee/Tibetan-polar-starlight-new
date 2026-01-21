#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
藏族文化 AI 绘画助手 - 主入口
"""

import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings

def setup_logging(level: str = "INFO"):
    """配置日志"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                settings.log_dir / "app.log",
                encoding='utf-8'
            )
        ]
    )

def check_dependencies():
    """检查依赖"""
    missing = []
    
    try:
        import gradio
    except ImportError:
        missing.append("gradio")
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")
    
    if missing:
        print(f"❌ 缺少依赖: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ 依赖检查通过")

def init_knowledge_base():
    """初始化知识库"""
    from middleware.rag_engine import create_rag_engine
    from middleware.ollama_client import get_ollama_client
    
    print("📚 初始化知识库...")
    
    ollama_client = get_ollama_client(
        base_url=settings.ollama_url,
        model=settings.ollama_model
    )
    
    rag_engine = create_rag_engine(
        persist_directory=settings.rag_persist_dir,
        ollama_client=ollama_client
    )
    
    # 检查是否需要导入默认知识
    stats = rag_engine.get_stats()
    if stats['document_count'] == 0:
        knowledge_file = settings.knowledge_base_dir / "tibetan_culture.txt"
        if knowledge_file.exists():
            print(f"📖 导入默认知识库: {knowledge_file}")
            rag_engine.add_documents_from_file(
                knowledge_file,
                metadata={"source": "default_knowledge_base"}
            )
            print(f"✅ 知识库初始化完成，共 {rag_engine.get_stats()['document_count']} 个文档块")
        else:
            print("⚠️ 默认知识库文件不存在，跳过导入")
    else:
        print(f"✅ 知识库已存在，共 {stats['document_count']} 个文档块")

def run_web_ui():
    """运行 Web 界面"""
    from ui.gradio_app import main as gradio_main
    gradio_main()

def run_cli():
    """运行命令行界面"""
    from cli.commands import main as cli_main
    cli_main()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="藏族文化 AI 绘画助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                    # 启动 Web 界面
  python main.py --cli              # 启动命令行界面
  python main.py --init-kb# 初始化知识库
  python main.py --check# 检查服务状态
        """
    )
    
    parser.add_argument(
        '--cli',
        action='store_true',
        help='使用命令行界面'
    )
    
    parser.add_argument(
        '--init-kb',
        action='store_true',
        help='初始化知识库'
    )
    
    parser.add_argument(
        '--check',
        action='store_true',
        help='检查服务状态'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='Web 服务器地址'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help='Web 服务器端口'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='创建公共分享链接'
    )
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                ║
    ║        🏔️  藏族文化 AI 绘画助手  🏔️                      ║
    ║                                   Tibetan Culture AI Art Assistant                       ║
    ║   基于 Stable Diffusion + Ollama + RAG                   ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # 检查依赖
    check_dependencies()
    
    # 检查服务状态
    if args.check:
        print("\n🔍 检查服务状态...\n")
        
        from middleware.sd_client import get_sd_client
        from middleware.ollama_client import get_ollama_client
        
        # 检查 SD WebUI
        sd_client = get_sd_client(base_url=settings.sd_webui_url)
        if sd_client.check_connection():
            print(f"✅ SD WebUI: 已连接 ({settings.sd_webui_url})")
        else:
            print(f"❌ SD WebUI: 未连接 ({settings.sd_webui_url})")
        
        # 检查 Ollama
        ollama_client = get_ollama_client(
            base_url=settings.ollama_url,
            model=settings.ollama_model
        )
        if ollama_client.check_connection():
            print(f"✅ Ollama: 已连接 ({settings.ollama_url})")
            models = ollama_client.list_models()
            if models:
                print(f"   可用模型: {', '.join(models[:5])}")
        else:
            print(f"❌ Ollama: 未连接 ({settings.ollama_url})")
        
        return
    
    # 初始化知识库
    if args.init_kb:
        init_knowledge_base()
        return
    
    # 更新设置
    if args.host:
        settings.gradio_host = args.host
    if args.port:
        settings.gradio_port = args.port
    if args.share:
        settings.gradio_share = True
    
    # 确保必要目录存在
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.rag_persist_dir.mkdir(parents=True, exist_ok=True)
    
    # 自动初始化知识库
    try:
        init_knowledge_base()
    except Exception as e:
        logger.warning(f"知识库初始化失败: {e}")
    
    # 启动界面
    if args.cli:
        print("\n🖥️ 启动命令行界面...\n")
        run_cli()
    else:
        print(f"\n🌐 启动 Web 界面...")
        print(f"   地址: http://{settings.gradio_host}:{settings.gradio_port}")
        print(f"   分享: {'是' if settings.gradio_share else '否'}\n")
        run_web_ui()

if __name__ == "__main__":
    main()