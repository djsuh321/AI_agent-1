from dotenv import load_dotenv
import os
import json

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")
MAX_STEPS = int(os.getenv("MAX_STEPS", 10))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 64000))
MAX_TURN_TOKENS = int(os.getenv("MAX_TURN_TOKENS", 2048))
TIMEOUT = int(os.getenv("TIMEOUT", 60))

HISTORY_FILE = "chat_history.json"

from llm_client import call_llm
from prompt_builder import get_system_prompt
from parser import parse_action
from tools import TOOL_REGISTRY
from utils import num_tokens_from_messages, setup_logger, trim_messages

logger = setup_logger()


def load_history(system_prompt):
    """从文件加载历史对话，若文件不存在则返回初始消息列表"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history and history[0]["role"] == "system":
                history[0]["content"] = system_prompt
            print(f"已加载历史对话（共 {len(history)-1} 条消息）")
            return history
        except Exception as e:
            print(f"历史记录加载失败，重新开始：{e}")
    return [{"role": "system", "content": system_prompt}]


def save_history(messages):
    """保存对话历史到文件"""
    try:
        filtered = [m for m in messages if not (
            m["role"] == "tool" or
            (m["role"] == "user" and "请严格按照格式输出" in m["content"])
        )]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        print(f"对话历史已保存到 {HISTORY_FILE}")
    except Exception as e:
        print(f"保存历史失败：{e}")


def agent_chat(user_input: str) -> str:
    """供web界面调用的单次对话接口"""
    system_prompt = get_system_prompt(TOOL_REGISTRY)
    messages = load_history(system_prompt)
    messages.append({"role": "user", "content": user_input})
    
    for step in range(MAX_STEPS):
        response = call_llm(messages, timeout=TIMEOUT, max_tokens=MAX_TURN_TOKENS)
        messages.append({"role": "assistant", "content": response})
        action_name, action_args = parse_action(response)
        
        if action_name is None:
            messages.append({"role": "user", "content": "请严格按照格式输出 Action 或 Final Answer。"})
            continue
        if action_name == "final_answer":
            save_history(messages)
            return action_args["answer"]
        if action_name in TOOL_REGISTRY:
            tool_result = TOOL_REGISTRY[action_name](**action_args)
            messages.append({"role": "tool", "content": f"{action_name} result: {tool_result}"})
            messages = trim_messages(messages, MAX_CONTEXT_TOKENS)
        
    save_history(messages)
    return "抱歉，未能在步骤限制内给出答案。"


def main():
    system_prompt = get_system_prompt(TOOL_REGISTRY)
    messages = load_history(system_prompt)

    try:
        while True:
            messages_in_one_turn = messages.copy()
            input_text = input("\n请输入您的问题（或输入 'exit' 退出）：")
            if input_text.lower() == "exit":
                print("退出程序。")
                break

            messages_in_one_turn.append({"role": "user", "content": input_text})

            for step in range(MAX_STEPS):
                logger.info(f"Step {step+1}/{MAX_STEPS}")

                response = call_llm(messages_in_one_turn, timeout=TIMEOUT, max_tokens=MAX_TURN_TOKENS)
                messages_in_one_turn.append({"role": "assistant", "content": response})
                logger.info(f"LLM Response: {response}")

                action_name, action_args = parse_action(response)

                if action_name is None:
                    logger.warning("LLM 输出格式错误，要求重新输出")
                    messages_in_one_turn.append({"role": "user", "content": "请严格按照格式输出 Action 或 Final Answer。"})
                    continue

                if action_name == "final_answer":
                    final_answer = action_args["answer"]
                    logger.info(f"Final Answer: {final_answer}")
                    print(f"\n助手最终答案：{final_answer}")
                    messages = messages_in_one_turn.copy()
                    break

                if action_name in TOOL_REGISTRY:
                    tool_func = TOOL_REGISTRY[action_name]
                    tool_result = tool_func(**action_args)
                    messages_in_one_turn.append({"role": "tool", "content": f"{action_name} result: {tool_result}"})
                    logger.info(f"Executed {action_name} with args {action_args}, got result: {tool_result}")
                else:
                    logger.warning(f"Unknown action: {action_name}")

                messages = trim_messages(messages_in_one_turn, MAX_CONTEXT_TOKENS)
            messages = messages_in_one_turn.copy()

    finally:
        save_history(messages)


if __name__ == "__main__":
    main()
