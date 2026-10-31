"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(usage: types.GenerateContentResponseUsageMetadata | None) -> None:
    """レスポンスに含まれるトークン使用量を表示する。"""
    if usage is None:
        print("[トークン] 使用量の情報が返ってきませんでした\n")
        return

    # 各フィールドは値が入っていないと None になるので、0 に置き換えておく
    input_tokens = usage.prompt_token_count or 0  # 入力（プロンプト）
    output_tokens = usage.candidates_token_count or 0  # 出力（回答本文）
    thoughts_tokens = usage.thoughts_token_count or 0  # 思考（思考するモデルのみ）

    print(f"[入力トークン] {input_tokens}")
    print(f"[出力トークン] {output_tokens}")
    if thoughts_tokens:
        print(f"[思考トークン] {thoughts_tokens}")
    print(f"[合計トークン] {usage.total_token_count}\n")


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    # 第1回 演習: 入力トークン数と出力トークン数を表示する
    print_usage(response.usage_metadata)

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()