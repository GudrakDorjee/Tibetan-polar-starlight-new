#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
命令行界面
提供简单的命令行交互方式
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from middleware.sd_client import get_sd_client, SDClient
from middleware.ollama_client import get_ollama_client, OllamaClient
from middleware.prompt_engineer import create_prompt_engineer, PromptEngineer
from middleware.rag_engine import create_rag_engine, RAGEngine

logger = logging.getLogger(__name__)

class TibetanArtCLI:
    """命令行界面类"""
    
    def __init__(self):
        self.sd_client: Optional[SDClient] = None
        self.ollama_client: Optional[OllamaClient] = None
        self.prompt_engineer: Optional[PromptEngineer] = None
        self.rag_engine: Optional[RAGEngine] = None
        
        self._initialize()
    
    def _initialize(self):
        """初始化客户端"""
        print("🔄 初始化中...")
        
        # SD WebUI
        self.sd_client = get_sd_client(
            base_url=settings.sd_webui_url,
            timeout=settings.generation_timeout
        )
        
        # Ollama
        self.ollama_client = get_ollama_client(
            base_url=settings.ollama_url,
            model=settings.ollama_model
        )
        
        # RAG
        self.rag_engine = create_rag_engine(
            persist_directory=settings.rag_persist_dir,
            ollama_client=self.ollama_client
        )
        
        # Prompt Engineer
        self.prompt_engineer = create_prompt_engineer(
            ollama_client=self.ollama_client,
            rag_engine=self.rag_engine
        )
        
        print("✅ 初始化完成")
    
    def check_status(self):
        """检查服务状态"""
        print("\n📊 服务状态:")
        print("-" * 40)
        
        # SD WebUI
        if self.sd_client and self.sd_client.check_connection():
            print(f"✅ SD WebUI: 已连接")
        else:
            print(f"❌ SD WebUI: 未连接")
        
        # Ollama
        if self.ollama_client and self.ollama_client.check_connection():
            print(f"✅ Ollama: 已连接")
        else:
            print(f"❌ Ollama: 未连接")
        
        # RAG
        if self.rag_engine:
            stats = self.rag_engine.get_stats()
            print(f"✅ RAG: {stats['document_count']} 文档")
        else:
            print(f"❌ RAG: 未初始化")
        
        print("-" * 40)
    
    def generate(
        self,
        prompt: str,
        output: Optional[str] = None,
        style: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        cfg_scale: float = 7.0,
        seed: int = -1,
        no_rag: bool = False,
        no_llm: bool = False
    ):
        """生成图片"""
        if not self.sd_client or not self.sd_client.check_connection():
            print("❌ SD WebUI 未连接，无法生成图片")
            return
        
        print(f"\n🎨 生成图片...")
        print(f"   描述: {prompt}")
        
        # 处理提示词
        if self.prompt_engineer:
            print("   处理提示词中...")
            result = self.prompt_engineer.process(
                user_input=prompt,
                style=style,
                use_rag=not no_rag,
                use_llm=not no_llm
            )
            positive_prompt = result.positive_prompt
            negative_prompt = result.negative_prompt
            
            print(f"   检测风格: {result.detected_style or '无'}")
            print(f"   检测关键词: {', '.join(result.detected_keywords) if result.detected_keywords else '无'}")
        else:
            positive_prompt = prompt
            negative_prompt = settings.default_negative_prompt
        
        print(f"\n   正向提示词: {positive_prompt[:100]}...")
        print(f"   负向提示词: {negative_prompt[:50]}...")
        
        # 生成图片
        print(f"\n   生成中 (尺寸: {width}x{height}, 步数: {steps})...")
        
        try:
            gen_result = self.sd_client.txt2img(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed
            )
            
            if gen_result.images:
                # 保存图片
                if output:
                    output_path = Path(output)
                else:
                    import time
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = settings.output_dir / f"tibetan_art_{timestamp}.png"
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                gen_result.images[0].save(output_path)

                print(f"\n✅ 生成成功!")
                print(f"   种子: {gen_result.seed}")
                print(f"   保存到: {output_path}")
            else:
                print("❌ 生成失败: 未返回图片")
        
        except TimeoutError as e:
            print(f"❌ 超时: {e}")
        except ConnectionError as e:
            print(f"❌ 连接错误: {e}")
        except Exception as e:
            print(f"❌ 错误: {e}")
            logger.exception("生成图片时发生错误")
    
    def translate(self, text: str):
        """翻译并优化提示词"""
        if not self.prompt_engineer:
            print("❌ Prompt 编排器未初始化")
            return
        
        print(f"\n🔄 处理提示词: {text}")
        print("-" * 40)
        
        result = self.prompt_engineer.process(
            user_input=text,
            use_rag=True,
            use_llm=True
        )
        
        print(f"\n📝 处理结果:")
        print(f"   检测风格: {result.detected_style or '无'}")
        print(f"   检测关键词: {', '.join(result.detected_keywords) if result.detected_keywords else '无'}")
        print(f"\n   正向提示词:")
        print(f"   {result.positive_prompt}")
        print(f"\n   负向提示词:")
        print(f"   {result.negative_prompt}")
    
    def query_knowledge(self, query: str, top_k: int = 3):
        """查询知识库"""
        if not self.rag_engine:
            print("❌ RAG 引擎未初始化")
            return
        
        print(f"\n🔍 查询: {query}")
        print("-" * 40)
        
        results = self.rag_engine.query(query, top_k=top_k)
        
        if not results:
            print("未找到相关内容")
            return
        
        for i, r in enumerate(results, 1):
            print(f"\n【结果 {i}】相似度: {r['score']:.3f}")
            print(f"来源: {r['metadata'].get('source', '未知')}")
            print(f"内容: {r['content']}")
            print("-" * 40)
    
    def list_models(self):
        """列出可用模型"""
        if not self.sd_client or not self.sd_client.check_connection():
            print("❌ SD WebUI 未连接")
            return
        
        print("\n📦 可用模型:")
        print("-" * 40)
        
        # Checkpoints
        checkpoints = self.sd_client.get_models()
        print(f"\n🎯 Checkpoints ({len(checkpoints)}):")
        for ckpt in checkpoints[:10]:
            print(f"   - {ckpt}")
        if len(checkpoints) > 10:
            print(f"   ... 还有 {len(checkpoints) - 10} 个")
        
        # LoRAs
        loras = self.sd_client.get_loras()
        print(f"\n🔧 LoRAs ({len(loras)}):")
        for lora in loras[:10]:
            print(f"   - {lora}")
        if len(loras) > 10:
            print(f"   ... 还有 {len(loras) - 10} 个")
        
        # Samplers
        samplers = self.sd_client.get_samplers()
        print(f"\n⚙️ 采样器 ({len(samplers)}):")
        for sampler in samplers:
            print(f"   - {sampler}")
    
    def interactive_mode(self):
        """交互模式"""
        print("\n🎮 进入交互模式 (输入 'help' 查看帮助, 'quit' 退出)")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n🏔️ > ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见!")
                    break
                
                if user_input.lower() == 'help':
                    self._print_interactive_help()
                    continue
                
                if user_input.lower() == 'status':
                    self.check_status()
                    continue
                
                if user_input.lower() == 'models':
                    self.list_models()
                    continue
                
                if user_input.lower().startswith('query '):
                    query = user_input[6:].strip()
                    self.query_knowledge(query)
                    continue
                
                if user_input.lower().startswith('translate '):
                    text = user_input[10:].strip()
                    self.translate(text)
                    continue
                
                # 默认作为生成命令
                self.generate(user_input)
            
            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
    def _print_interactive_help(self):
        """打印交互模式帮助"""
        print("""
📖 交互模式命令:

  <描述>              直接输入描述生成图片
  translate <文本>    翻译并优化提示词
  query <关键词>      查询知识库
  status              查看服务状态
  models              列出可用模型
  help                显示此帮助
  quit / exit / q     退出

📝 示例:
  > 一位康巴姑娘站在草原上
  > translate 格萨尔王骑马
  > query 唐卡
        """)

def main():
    """CLI 主函数"""
    parser = argparse.ArgumentParser(
        description="藏族文化 AI 绘画助手 - 命令行界面",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # generate 命令
    gen_parser = subparsers.add_parser('generate', aliases=['gen', 'g'], help='生成图片')
    gen_parser.add_argument('prompt', type=str, help='图片描述')
    gen_parser.add_argument('-o', '--output', type=str, help='输出文件路径')
    gen_parser.add_argument('-s', '--style', type=str, help='风格预设')
    gen_parser.add_argument('-W', '--width', type=int, default=512, help='图片宽度')
    gen_parser.add_argument('-H', '--height', type=int, default=512, help='图片高度')
    gen_parser.add_argument('--steps', type=int, default=30, help='采样步数')
    gen_parser.add_argument('--cfg', type=float, default=7.0, help='CFG Scale')
    gen_parser.add_argument('--seed', type=int, default=-1, help='随机种子')
    gen_parser.add_argument('--no-rag', action='store_true', help='禁用知识库增强')
    gen_parser.add_argument('--no-llm', action='store_true', help='禁用 LLM 翻译')
    
    # translate 命令
    trans_parser = subparsers.add_parser('translate', aliases=['trans', 't'], help='翻译提示词')
    trans_parser.add_argument('text', type=str, help='要翻译的文本')
    
    # query 命令
    query_parser = subparsers.add_parser('query', aliases=['q'], help='查询知识库')
    query_parser.add_argument('query', type=str, help='查询内容')
    query_parser.add_argument('-k', '--top-k', type=int, default=3, help='返回结果数量')
    
    # status 命令
    subparsers.add_parser('status', help='查看服务状态')
    
    # models 命令
    subparsers.add_parser('models', help='列出可用模型')
    
    # interactive 命令
    subparsers.add_parser('interactive', aliases=['i'], help='进入交互模式')
    
    args = parser.parse_args()
    
    # 创建 CLI 实例
    cli = TibetanArtCLI()
    
    # 执行命令
    if args.command in ['generate', 'gen', 'g']:
        cli.generate(
            prompt=args.prompt,
            output=args.output,
            style=args.style,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg_scale=args.cfg,
            seed=args.seed,
            no_rag=args.no_rag,
            no_llm=args.no_llm
        )
    
    elif args.command in ['translate', 'trans', 't']:
        cli.translate(args.text)
    
    elif args.command in ['query', 'q']:
        cli.query_knowledge(args.query, args.top_k)
    
    elif args.command == 'status':
        cli.check_status()
    
    elif args.command == 'models':
        cli.list_models()
    
    elif args.command in ['interactive', 'i']:
        cli.interactive_mode()
    
    else:
        # 默认进入交互模式
        cli.interactive_mode()

if __name__ == "__main__":
    main()