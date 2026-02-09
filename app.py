# app.py
import os
import json
import time
from datetime import datetime, timedelta

import requests
import pandas as pd
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")


# -----------------------------
# Utilities / API
# -----------------------------
def safe_get_json(url: str, params=None, headers=None, timeout=10):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def get_weather(city_query: str, api_key: str):
    """
    OpenWeatherMap에서 날씨 가져오기 (한국어, 섭씨)
    실패 시 None 반환, timeout=10
    """
    if not api_key:
        return None

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_query,
        "appid": api_key,
        "units": "metric",
        "lang": "kr",
    }
    data = safe_get_json(url, params=params, timeout=10)
    if not data:
        return None

    try:
        weather = {
            "city": data.get("name"),
            "desc": (data.get("weather") or [{}])[0].get("description"),
            "temp": (data.get("main") or {}).get("temp"),
            "feels_like": (data.get("main") or {}).get("feels_like"),
            "humidity": (data.get("main") or {}).get("humidity"),
            "wind": (data.get("wind") or {}).get("speed"),
        }
        # 필수값 없으면 None
        if weather["desc"] is None or weather["temp"] is None:
            return None
        return weather
    except Exception:
        return None


def get_dog_image():
    """
    Dog CEO에서 랜덤 강아지 사진 URL과 품종 가져오기
    실패 시 None 반환, timeout=10
    """
    url = "https://dog.ceo/api/breeds/image/random"
    data = safe_get_json(url, timeout=10)
    if not data or data.get("status") != "success":
        return None

    try:
        img_url = data.get("message")
        if not img_url:
            return None

        # URL에서 품종 추정: .../breeds/{breed}[-subbreed]/...
        breed = "알 수 없음"
        parts = img_url.split("/breeds/")
        if len(parts) > 1:
            breed_part = parts[1].split("/")[0]  # e.g., hound-afghan
            breed_part = breed_part.replace("-", " ")
            breed = breed_part.strip() if breed_part.strip() else "알 수 없음"

        return {"url": img_url, "breed": breed}
    except Exception:
        return None


def _openai_chat_completion(openai_api_key: str, model: str, system: str, user: str, timeout=20):
    """
    OpenAI 호출 (가능하면 공식 SDK 사용, 실패하면 REST로 폴백)
    """
    if not openai_api_key:
        return None

    # 1) Official SDK (new)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception:
        pass

    # 2) REST fallback
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def generate_report(
    openai_api_key: str,
    coach_style: str,
    date_str: str,
    habits_checked: list,
    mood: int,
    weather: dict | None,
    dog: dict | None,
):
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달
    코치 스타일별 시스템 프롬프트 (스파르타=엄격, 멘토=따뜻, 게임마스터=RPG)
    출력 형식: 컨디션 등급(S~D), 습관 분석, 날씨 코멘트, 내일 미션, 오늘의 한마디
    모델: gpt-5-mini
    """
    style_prompts = {
        "스파르타 코치": (
            "너는 매우 엄격하고 직설적인 '스파르타 코치'다. "
            "핑계는 허용하지 않지만, 실행 가능한 지시를 준다. "
            "반드시 한국어로 간결하고 단호하게 말해라."
        ),
        "따뜻한 멘토": (
            "너는 공감이 뛰어난 '따뜻한 멘토'다. "
            "사용자의 감정을 존중하고, 부담이 적은 다음 행동을 제안한다. "
            "반드시 한국어로 다정하고 명확하게 말해라."
        ),
        "게임 마스터": (
            "너는 RPG 세계관의 '게임 마스터'다. "
            "사용자를 모험가로 설정하고 퀘스트/보상/레벨업 같은 표현을 사용한다. "
            "너무 과하지 않게, 하지만 재미있게. 반드시 한국어로 말해라."
        ),
    }

    system = style_prompts.get(coach_style, style_prompts["따뜻한 멘토"])

    # 입력 요약(LLM에 전달)
    payload = {
        "date": date_str,
        "habits_checked": habits_checked,
        "habits_count": len(habits_checked),
        "habits_total": 5,
        "mood_1_to_10": mood,
        "weather": weather or None,
        "dog": dog or None,
        "output_format": {
            "컨디션 등급": "S/A/B/C/D 중 하나",
            "습관 분석": "잘한 점 + 아쉬운 점 + 한 문장 요약",
            "날씨 코멘트": "날씨가 습관/컨디션에 미치는 영향과 팁",
            "내일 미션": "3개, 체크박스 형태로(예: - [ ] ...)",
            "오늘의 한마디": "짧고 임팩트 있게",
        },
    }

    user = (
        "아래 사용자 데이터를 기반으로 'AI 습관 트래커' 리포트를 작성해줘.\n"
        "반드시 다음 섹션 헤더를 그대로 사용해서 출력해:\n"
        "1) 컨디션 등급\n"
        "2) 습관 분석\n"
        "3) 날씨 코멘트\n"
        "4) 내일 미션\n"
        "5) 오늘의 한마디\n\n"
        "컨디션 등급은 반드시 S/A/B/C/D 중 하나로만.\n"
        "내일 미션은 반드시 3개, 체크박스 형식(- [ ] )으로.\n\n"
        f"사용자 데이터(JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    return _openai_chat_completion(
        openai_api_key=openai_api_key,
        model="gpt-5-mini",
        system=system,
        user=user,
        timeout=25,
    )


# -----------------------------
# Session State Init
# -----------------------------
if "history" not in st.session_state:
    # 기록: { "YYYY-MM-DD": {"habits": [...], "mood": int, "city": str, "style": str, "rate": float} }
    st.session_state["history"] = {}

if "demo_seeded" not in st.session_state:
    st.session_state["demo_seeded"] = False

if not st.session_state["demo_seeded"]:
    # 데모용 6일 샘플 데이터
    today = datetime.now().date()
    habit_names = ["기상 미션", "물 마시기", "공부/독서", "운동하기", "수면"]
    # 간단한 패턴으로 샘플 생성
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        checked = [h for idx, h in enumerate(habit_names) if (idx + i) % 2 == 0]
        mood = max(1, min(10, 4 + (i % 7)))
        rate = round((len(checked) / 5) * 100, 1)
        st.session_state["history"][d.isoformat()] = {
            "habits": checked,
            "mood": mood,
            "city": "Seoul",
            "style": "따뜻한 멘토",
            "rate": rate,
        }
    st.session_state["demo_seeded"] = True


# -----------------------------
# Sidebar: API Keys
# -----------------------------
with st.sidebar:
    st.header("🔑 API 설정")
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    weather_key = st.text_input("OpenWeatherMap API Key", type="password", value=os.getenv("OPENWEATHER_API_KEY", ""))
    st.caption("키는 브라우저 세션에서만 사용되며, 서버에 저장되지 않도록 구성하는 것을 권장합니다.")


# -----------------------------
# Main UI
# -----------------------------
st.title("📊 AI 습관 트래커")
st.caption("오늘의 습관을 체크하고, 날씨 + 강아지 + AI 코치 리포트로 컨디션을 점검해요.")

# 도시 선택 (10개) + 코치 스타일
CITY_OPTIONS = {
    "Seoul": "Seoul,KR",
    "Busan": "Busan,KR",
    "Incheon": "Incheon,KR",
    "Daegu": "Daegu,KR",
    "Daejeon": "Daejeon,KR",
    "Gwangju": "Gwangju,KR",
    "Suwon": "Suwon,KR",
    "Ulsan": "Ulsan,KR",
    "Jeju": "Jeju City,KR",
    "Gangneung": "Gangneung,KR",
}
COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]

top_left, top_right = st.columns([1, 1])

with top_left:
    st.subheader("✅ 습관 체크인")
    # 체크박스 5개를 2열로 배치 + 이모지
    habits = [
        ("🌅", "기상 미션"),
        ("💧", "물 마시기"),
        ("📚", "공부/독서"),
        ("🏃", "운동하기"),
        ("😴", "수면"),
    ]

    c1, c2 = st.columns(2)
    checked = []
    for idx, (emo, label) in enumerate(habits):
        col = c1 if idx % 2 == 0 else c2
        with col:
            if st.checkbox(f"{emo} {label}", key=f"habit_{label}"):
                checked.append(label)

    mood = st.slider("🙂 오늘 기분은 어때요? (1~10)", min_value=1, max_value=10, value=6)

with top_right:
    st.subheader("🌍 환경 설정")
    city_display = st.selectbox("도시 선택", list(CITY_OPTIONS.keys()), index=0)
    coach_style = st.radio("코치 스타일", COACH_STYLES, horizontal=True, index=1)

# 달성률 계산
total = 5
done = len(checked)
rate = round((done / total) * 100, 1)

m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{rate}%", help="체크된 습관 수 / 5")
m2.metric("달성 습관", f"{done} / {total}")
m3.metric("기분", f"{mood} / 10")

st.divider()

# -----------------------------
# Save today's record to session_state (on change-like behavior)
# -----------------------------
today_str = datetime.now().date().isoformat()
st.session_state["history"][today_str] = {
    "habits": checked,
    "mood": mood,
    "city": city_display,
    "style": coach_style,
    "rate": rate,
}

# -----------------------------
# 7-day Bar Chart (6 demo + today)
# -----------------------------
st.subheader("📅 최근 7일 달성률")

# 최근 7일 날짜 정렬
dates = [datetime.now().date() - timedelta(days=i) for i in range(6, -1, -1)]
rows = []
for d in dates:
    key = d.isoformat()
    rec = st.session_state["history"].get(key)
    rows.append(
        {
            "date": key,
            "달성률(%)": rec["rate"] if rec else 0.0,
            "달성 습관 수": len(rec["habits"]) if rec else 0,
            "기분": rec["mood"] if rec else 0,
        }
    )

df = pd.DataFrame(rows)
df_display = df.copy()
df_display["date"] = pd.to_datetime(df_display["date"]).dt.strftime("%m/%d")

chart_cols = st.columns([2, 1])
with chart_cols[0]:
    st.bar_chart(df_display.set_index("date")["달성률(%)"])
with chart_cols[1]:
    st.dataframe(df_display, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# Report Generation
# -----------------------------
st.subheader("🧠 AI 코치 리포트")

btn_cols = st.columns([1, 3])
with btn_cols[0]:
    gen = st.button("컨디션 리포트 생성", use_container_width=True)

# API 결과/리포트는 버튼 눌렀을 때만 갱신
if "latest_weather" not in st.session_state:
    st.session_state["latest_weather"] = None
if "latest_dog" not in st.session_state:
    st.session_state["latest_dog"] = None
if "latest_report" not in st.session_state:
    st.session_state["latest_report"] = None
if "latest_share" not in st.session_state:
    st.session_state["latest_share"] = ""

if gen:
    # 1) Weather
    weather = get_weather(CITY_OPTIONS[city_display], weather_key)
    st.session_state["latest_weather"] = weather

    # 2) Dog
    dog = get_dog_image()
    st.session_state["latest_dog"] = dog

    # 3) Report
    report = generate_report(
        openai_api_key=openai_key,
        coach_style=coach_style,
        date_str=today_str,
        habits_checked=checked,
        mood=mood,
        weather=weather,
        dog=dog,
    )
    st.session_state["latest_report"] = report

    # 4) Share text
    weather_line = (
        f"날씨: {weather['desc']} / {weather['temp']}°C (체감 {weather['feels_like']}°C)"
        if weather
        else "날씨: (가져오지 못했어요)"
    )
    dog_line = f"강아지: {dog['breed']}" if dog else "강아지: (가져오지 못했어요)"
    habits_line = " / ".join(checked) if checked else "아직 체크한 습관이 없어요"

    share = (
        f"📊 AI 습관 트래커 ({today_str})\n"
        f"✅ 달성률: {rate}% ({done}/{total})\n"
        f"🧩 습관: {habits_line}\n"
        f"🙂 기분: {mood}/10\n"
        f"🌍 도시: {city_display}\n"
        f"{weather_line}\n"
        f"{dog_line}\n"
        f"🧠 코치 스타일: {coach_style}\n"
    )
    st.session_state["latest_share"] = share

# -----------------------------
# Results Display: Weather + Dog card (2 columns) + AI Report
# -----------------------------
res_left, res_right = st.columns(2)

with res_left:
    st.markdown("#### 🌦️ 오늘의 날씨")
    w = st.session_state["latest_weather"]
    if w:
        st.info(
            f"**{w['city']}**\n\n"
            f"- 상태: **{w['desc']}**\n"
            f"- 기온: **{w['temp']}°C** (체감 {w['feels_like']}°C)\n"
            f"- 습도: {w['humidity']}%\n"
            f"- 바람: {w['wind']} m/s"
        )
    else:
        st.warning("날씨 정보를 아직 불러오지 않았거나, 가져오지 못했어요. (API Key/도시/네트워크 확인)")

with res_right:
    st.markdown("#### 🐶 오늘의 강아지")
    d = st.session_state["latest_dog"]
    if d:
        st.image(d["url"], use_container_width=True, caption=f"품종(추정): {d['breed']}")
    else:
        st.warning("강아지 이미지를 아직 불러오지 않았거나, 가져오지 못했어요. (네트워크 확인)")

st.markdown("#### 📝 AI 코치 리포트")
rep = st.session_state["latest_report"]
if rep:
    st.markdown(rep)
else:
    st.caption("버튼을 눌러 리포트를 생성해보세요. (OpenAI API Key 필요)")

st.markdown("#### 📣 공유용 텍스트")
st.code(st.session_state.get("latest_share", ""), language="text")

# -----------------------------
# Footer: API 안내 (expander)
# -----------------------------
with st.expander("ℹ️ API 안내 / 설정 팁"):
    st.markdown(
        """
- **OpenAI API Key**
  - AI 코치 리포트를 생성할 때 사용됩니다.
  - 이 앱은 기본적으로 **모델: `gpt-5-mini`** 를 호출합니다.
- **OpenWeatherMap API Key**
  - 도시의 현재 날씨를 가져옵니다. (한국어, 섭씨)
  - 동작 확인이 필요하면 먼저 키가 유효한지 OpenWeatherMap 콘솔에서 테스트해보세요.
- **Dog CEO API**
  - 무료 공개 API로 랜덤 강아지 이미지를 제공합니다.

**문제 해결 체크리스트**
- 키가 비어있지 않은지 확인
- OpenWeatherMap은 무료 플랜에서 호출 제한/권한 설정이 있을 수 있어요.
- 네트워크(사내망/방화벽)에서 외부 API 호출이 막힐 수 있습니다.
"""
    )
