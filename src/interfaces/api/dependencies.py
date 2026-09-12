from fastapi import Header, HTTPException

# 暫定IDの接頭辞。認証追加後は "oidc:" に変わる（仕様書7.4）
USER_ID_PREFIX = "local:"


def get_current_user_id(x_user_id: str = Header()) -> str:
    """リクエストから利用者IDを取り出す。利用者の識別方法を知っている唯一の関数。
    認証追加時は、この関数だけをJWTの sub を読む実装に差し替える（仕様書7.4）"""
    name = x_user_id.strip()
    if not name:
        raise HTTPException(status_code=400, detail="X-User-Id ヘッダが必要です")
    return f"{USER_ID_PREFIX}{name}"
