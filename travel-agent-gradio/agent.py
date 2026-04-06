"""
travel_agent.py — A travel planning agent built on bare-bones (no libraries/frameworks) agent pattern.

Run it:
    pip install openai gradio pytz tavily
    export OPENAI_API_KEY=sk-...
    export TAVILY_API_KEY=tvly-...
    python3 agent.py

Tools:
    1. search_web              — real-time web search (Tavily) for anything current
    2. get_destination_info    — search for climate, best time, visa, highlights
    3. get_weather             — current + 5-day forecast for any city
    4. generate_itinerary      — structured day-by-day plan
    5. estimate_budget         — search for current costs, hostel/hotel prices
    6. get_destinations        — curated list filtered by vibe
"""

import os
import json
import gradio as gr
from openai import OpenAI

from tavily import TavilyClient


# =============================================================================
# PART 1: TOOLS
# =============================================================================

def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for up-to-date travel information.

    Use this for anything that changes over time:
    - current visa requirements
    - entry restrictions or travel advisories
    - recent hotel/flight prices
    - upcoming events or festivals
    - current exchange rates
    - recent traveller reviews

    Args:
        query: A specific search query, e.g. 'Tokyo visa requirements for Indian passport 2025'
        max_results: How many results to return (1-5). Default 5.
    """
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        result = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",     
            include_answer=True,        # Tavily provides a pre-summarised answer
        )

        # Build a clean summary from the results
        output_parts = []

        # Include Tavily's pre-written summary
        if result.get("answer"):
            output_parts.append(f"Summary: {result['answer']}")

        # Add individual source snippets
        for i, r in enumerate(result.get("results", [])[:max_results], 1):
            title   = r.get("title", "")
            content = r.get("content", "")[:400]   # trim long snippets
            url     = r.get("url", "")
            output_parts.append(f"\n[{i}] {title}\n{content}\nSource: {url}")

        return "\n".join(output_parts) if output_parts else "No results found."

    except Exception as e:
        return f"Search error: {e}"
    

def get_destinations(vibe: str = "any") -> str:
    """Get popular travel destinations filtered by the type of experience wanted.

    Args:
        vibe: 'beach', 'city', 'nature', 'culture', 'adventure', 'food', or 'any'.
    """
    destinations = {
        "beach":     ["Bali", "Maldives", "Santorini", "Phuket", "Cancun", "Amalfi Coast", "Fiji", "Zanzibar"],
        "city":      ["Tokyo", "New York City", "Paris", "London", "Singapore", "Barcelona", "Dubai", "Istanbul"],
        "nature":    ["Patagonia", "Iceland", "New Zealand", "Costa Rica", "Norwegian Fjords", "Banff", "Galápagos"],
        "culture":   ["Kyoto", "Cairo", "Rome", "Istanbul", "Marrakech", "Varanasi", "Cusco", "Petra"],
        "adventure": ["Nepal", "Queenstown", "Moab", "Interlaken", "Kilimanjaro", "Patagonia", "Borneo"],
        "food":      ["Tokyo", "Naples", "Barcelona", "Bangkok", "Lyon", "Mexico City", "Istanbul", "Osaka"],
    }
    if vibe == "any":
        all_places = sorted(set(p for places in destinations.values() for p in places))
        return f"Popular destinations: {', '.join(all_places)}"
    vibe = vibe.lower()
    if vibe in destinations:
        return f"Top {vibe} destinations: {', '.join(destinations[vibe])}"
    return f"Unknown vibe '{vibe}'. Options: beach, city, nature, culture, adventure, food, any."


def get_destination_info(destination: str) -> str:
    """Get comprehensive info about a destination by searching the web.

    Searches for: best time to visit, climate, top attractions, visa info,
    local tips, and safety considerations.

    Args:
        destination: City or country name, e.g. 'Kyoto Japan' or 'Lisbon Portugal'
    """
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        # Run two focused searches and combine
        general = client.search(
            query=f"{destination} travel guide best time to visit highlights tips",
            max_results=3,
            search_depth="basic",
            include_answer=True,
        )
        visa = client.search(
            query=f"{destination} visa requirements tourists 2025",
            max_results=2,
            search_depth="basic",
            include_answer=True,
        )

        parts = [f"=== {destination} Travel Info ==="]

        if general.get("answer"):
            parts.append(f"\nOverview:\n{general['answer']}")

        for r in general.get("results", [])[:3]:
            parts.append(f"\n• {r.get('title','')}: {r.get('content','')[:300]}")

        if visa.get("answer"):
            parts.append(f"\nVisa Info:\n{visa['answer']}")

        return "\n".join(parts)

    except Exception as e:
        return f"Search error: {e}"
    

def get_destination_info_static(destination: str) -> str:
    """Get key facts about a destination: best time to visit, climate, and highlights - from database

    Args:
        destination: The name of the destination, e.g. 'Tokyo' or 'Bali'.
    """
    info = {
        "tokyo": {
            "best_time": "March-May (cherry blossom) or Oct-Nov (autumn foliage)",
            "climate": "Humid subtropical. Hot summers, mild winters, rainy June.",
            "highlights": "Shibuya crossing, Senso-ji temple, Tsukiji market, Akihabara, day trips to Nikko or Hakone",
            "language": "Japanese (English signage in tourist areas)",
            "currency": "Japanese Yen (JPY)",
            "timezone": "JST (UTC+9)",
        },
        "bali": {
            "best_time": "April-October (dry season)",
            "climate": "Tropical. Dry season Apr-Oct, wet season Nov-Mar.",
            "highlights": "Ubud rice terraces, Uluwatu temple, Seminyak beach, Mount Batur sunrise hike",
            "language": "Balinese / Indonesian (English widely spoken in tourist areas)",
            "currency": "Indonesian Rupiah (IDR)",
            "timezone": "WITA (UTC+8)",
        },
        "paris": {
            "best_time": "April-June or September-October",
            "climate": "Oceanic. Mild, occasional rain. Hot summers, cold winters.",
            "highlights": "Eiffel Tower, Louvre, Montmartre, Seine river cruise, day trip to Versailles",
            "language": "French (English spoken in tourist areas)",
            "currency": "Euro (EUR)",
            "timezone": "CET (UTC+1)",
        },
        "barcelona": {
            "best_time": "May-June or September-October",
            "climate": "Mediterranean. Warm sunny summers, mild winters.",
            "highlights": "Sagrada Família, Park Güell, La Boqueria, Gothic Quarter, Barceloneta beach",
            "language": "Catalan / Spanish (English widely spoken)",
            "currency": "Euro (EUR)",
            "timezone": "CET (UTC+1)",
        },
        "new york city": {
            "best_time": "April-June or September-November",
            "climate": "Humid continental. Hot summers, cold winters, rain year-round.",
            "highlights": "Central Park, Times Square, MoMA, Brooklyn Bridge, The High Line, food in every neighborhood",
            "language": "English",
            "currency": "USD",
            "timezone": "EST (UTC-5)",
        },
        "cape town": {
            "best_time": "November-March (Southern Hemisphere summer)",
            "climate": "Mediterranean. Warm dry summers, mild wet winters.",
            "highlights": "Table Mountain, Cape of Good Hope, Boulders Beach penguins, Winelands day trip",
            "language": "English / Afrikaans / Xhosa",
            "currency": "South African Rand (ZAR)",
            "timezone": "SAST (UTC+2)",
        },
    }

    key = destination.lower().strip()
    if key in info:
        d = info[key]
        return (
            f"Destination: {destination}\n"
            f"Best time to visit: {d['best_time']}\n"
            f"Climate: {d['climate']}\n"
            f"Top highlights: {d['highlights']}\n"
            f"Language: {d['language']}\n"
            f"Currency: {d['currency']}\n"
            f"Timezone: {d['timezone']}"
        )

    # For destinations not in the local database, return a prompt for the model
    # to answer from its own knowledge or use web-search option for accuracy
    return (
        f"No pre-loaded data for '{destination}'. "
        f"Please use your own knowledge to describe this destination's best time to visit, "
        f"climate, and top highlights."
    )

def get_weather(city: str) -> str:
    """Get the current weather and a short forecast for a city.

    Uses Open-Meteo (free) for real weather data.

    Args:
        city: City name, e.g. 'Tokyo', 'Paris', 'Bali'
    """
    import urllib.request

    # Step 1: geocode the city name to lat/lon using Open-Meteo's geocoding API
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1"
        with urllib.request.urlopen(geo_url, timeout=5) as resp:
            geo_data = json.loads(resp.read())

        if not geo_data.get("results"):
            return f"Could not find location data for '{city}'."

        loc     = geo_data["results"][0]
        lat     = loc["latitude"]
        lon     = loc["longitude"]
        name    = loc.get("name", city)
        country = loc.get("country", "")

    except Exception as e:
        return f"Geocoding error for '{city}': {e}"

    # Step 2: fetch weather from Open-Meteo
    try:
        import urllib.parse
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"
            f"&timezone=auto&forecast_days=5"
        )
        with urllib.request.urlopen(weather_url, timeout=5) as resp:
            weather = json.loads(resp.read())

        current = weather.get("current", {})
        daily   = weather.get("daily", {})

        # WMO weather code to plain English
        def wmo_description(code):
            codes = {
                0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                45: "Foggy", 48: "Icy fog",
                51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
                61: "Light rain", 63: "Rain", 65: "Heavy rain",
                71: "Light snow", 73: "Snow", 75: "Heavy snow",
                80: "Light showers", 81: "Showers", 82: "Violent showers",
                95: "Thunderstorm", 99: "Thunderstorm with hail",
            }
            return codes.get(code, f"Code {code}")

        temp    = current.get("temperature_2m", "?")
        humidity = current.get("relative_humidity_2m", "?")
        wind    = current.get("wind_speed_10m", "?")
        cond    = wmo_description(current.get("weather_code", 0))

        lines = [
            f"Weather in {name}, {country}:",
            f"  Now: {temp}°C, {cond}, humidity {humidity}%, wind {wind} km/h",
            "\n5-day forecast:",
        ]

        dates   = daily.get("time", [])
        max_t   = daily.get("temperature_2m_max", [])
        min_t   = daily.get("temperature_2m_min", [])
        precip  = daily.get("precipitation_sum", [])
        d_codes = daily.get("weather_code", [])

        for i in range(min(5, len(dates))):
            lines.append(
                f"  {dates[i]}: {wmo_description(d_codes[i])}, "
                f"{min_t[i]}–{max_t[i]}°C, rain {precip[i]}mm"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Weather fetch error: {e}"
    

def generate_itinerary(destination: str, days: int, travel_style: str = "balanced",
                       interests: str = "") -> str:
    """Generate a structured day-by-day itinerary request.

    Args:
        destination: Where the user is travelling.
        days: Number of days (1-14).
        travel_style: 'relaxed' (2-3 things/day), 'balanced' (3-4), or 'packed' (5+).
        interests: Optional comma-separated interests, e.g. 'food, temples, hiking'.
    """
    if days < 1:
        return "Error: days must be at least 1."
    if days > 14:
        return "Error: itineraries longer than 14 days need more customisation — please ask me to plan it in segments."

    style_note = {
        "relaxed": "Leisurely pace — 2-3 activities per day, long lunches, room to wander.",
        "balanced": "Mix of sightseeing and downtime — 3-4 things per day.",
        "packed":   "Maximise every day — early starts, 5+ activities, efficient routing.",
    }.get(travel_style.lower(), "3-4 activities per day.")

    interest_note = f"\nFocus on: {interests}." if interests else ""

    return (
        f"ITINERARY PARAMETERS\n"
        f"Destination: {destination}\n"
        f"Duration: {days} days\n"
        f"Style: {travel_style} — {style_note}"
        f"{interest_note}\n\n"
        f"Write a complete day-by-day itinerary. For each day include:\n"
        f"  • Morning: specific activity + practical tip (opening hours, how to get there)\n"
        f"  • Lunch: specific restaurant or food market with a dish to try\n"
        f"  • Afternoon: activity or neighbourhood to explore\n"
        f"  • Dinner: specific restaurant recommendation with price range\n"
        f"  • Evening: optional activity or bar/area\n"
        f"  • Insider tip: one thing most tourists miss\n"
        f"Be specific with real place names. Don't be generic."
    )

def estimate_budget(destination: str, days: int, travel_style: str = "mid-range") -> str:
    """Search for current budget estimates for a destination.

    Args:
        destination: The destination city or country.
        days: Number of days.
        travel_style: 'budget', 'mid-range', or 'luxury'.
    """
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        result = client.search(
            query=f"{destination} travel cost per day {travel_style} budget 2025 accommodation food transport",
            max_results=4,
            search_depth="basic",
            include_answer=True,
        )

        parts = [f"Budget research for {days} days in {destination} ({travel_style}):"]

        if result.get("answer"):
            parts.append(f"\n{result['answer']}")

        for r in result.get("results", [])[:3]:
            parts.append(f"\n• {r.get('title','')}: {r.get('content','')[:300]}")

        parts.append(f"\nNote: multiply daily costs by {days} for your total estimate. Flights not included.")
        return "\n".join(parts)

    except Exception as e:
        return f"Search error: {e}"
    

def estimate_budget_static(destination: str, days: int, travel_style: str = "mid-range") -> str:
    """Estimate a rough daily and total budget for a trip - from database

    Args:
        destination: The destination city or country.
        days: Number of days.
        travel_style: Budget level — 'budget', 'mid-range', or 'luxury'.
    """
    # Cost index: rough USD per person per day (accommodation + food + transport + activities)
    cost_index = {
        # (budget, mid-range, luxury)
        "tokyo":         (80,  180, 400),
        "bali":          (40,  90,  250),
        "paris":         (100, 220, 500),
        "barcelona":     (80,  170, 380),
        "new york city": (120, 250, 600),
        "cape town":     (60,  130, 320),
        "london":        (110, 230, 550),
        "bangkok":       (35,  80,  200),
        "rome":          (80,  180, 420),
        "sydney":        (100, 210, 480),
    }

    style_map = {"budget": 0, "mid-range": 1, "luxury": 2}
    style_idx = style_map.get(travel_style.lower(), 1)

    key = destination.lower().strip()
    if key in cost_index:
        daily = cost_index[key][style_idx]
        total = daily * days
        breakdown = {
            "budget":    "hostels/guesthouses, street food, public transport, free sights",
            "mid-range": "3-star hotels, sit-down restaurants, mix of transport, paid attractions",
            "luxury":    "4-5 star hotels, fine dining, taxis/private transfers, premium experiences",
        }[travel_style.lower() if travel_style.lower() in style_map else "mid-range"]

        return (
            f"Budget estimate for {days} days in {destination} ({travel_style}):\n"
            f"  Daily: ~${daily} USD per person\n"
            f"  Total: ~${total} USD per person\n"
            f"  Includes: {breakdown}\n"
            f"  Note: Flights not included. Prices are rough estimates — book ahead to save."
        )

    return (
        f"No cost data pre-loaded for '{destination}'. "
        f"As a rough guide: budget travel in most of SE Asia runs $40-60/day, "
        f"Western Europe $100-200/day, and major US cities $150-250/day."
    )


def get_packing_list(destination: str, days: int, season: str) -> str:
    """Get a recommended packing list for a trip.

    Args:
        destination: Where the user is going.
        days: Length of the trip in days.
        season: Time of year — 'summer', 'winter', 'spring', or 'autumn'.
    """
    base = [
        "Passport + photocopies stored separately",
        "Travel insurance documents",
        "Phone + charger + universal adapter",
        "Portable battery pack",
        "Medications + basic first aid (plasters, paracetamol)",
        "Debit/credit cards + some local cash",
        "Reusable water bottle",
    ]

    seasonal = {
        "summer": ["Sunscreen SPF 50+", "Sunglasses", "Sun hat", "Light breathable clothing",
                   "Sandals", "Insect repellent (for tropical destinations)"],
        "winter": ["Warm coat", "Thermal underlayers", "Gloves + scarf + hat",
                   "Waterproof boots", "Hand warmers"],
        "spring": ["Light jacket (layers)", "Umbrella / packable rain jacket",
                   "Mix of warm and cool clothing"],
        "autumn": ["Medium-weight jacket", "Umbrella", "Layers for cool evenings"],
    }

    trip_length_note = ""
    if days <= 3:
        trip_length_note = "Short trip: a carry-on bag should be enough."
    elif days <= 7:
        trip_length_note = "Week-long trip: consider one checked bag or a large backpack."
    else:
        trip_length_note = "Long trip: pack light and plan to do laundry."

    season_key = season.lower().strip()
    season_items = seasonal.get(season_key, seasonal["summer"])

    all_items = base + season_items
    numbered = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(all_items))

    return (
        f"Packing list for {days} days in {destination} ({season}):\n"
        f"{numbered}\n\n"
        f"Tip: {trip_length_note}"
    )


TOOL_FUNCTIONS = {
    "search_web":           search_web,
    "get_destinations":     get_destinations,
    "get_destination_info": get_destination_info,
    "get_weather":          get_weather,
    "generate_itinerary":   generate_itinerary,
    "estimate_budget":      estimate_budget,
    "get_packing_list":     get_packing_list,
}


# =============================================================================
# PART 2: TOOL SCHEMAS
# =============================================================================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_destinations",
            "description": "Get a curated list of popular destinations filtered by travel vibe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vibe": {"type": "string", "description": "'beach', 'city', 'nature', 'culture', 'adventure', 'food', or 'any'."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for current travel information. Use for: visa requirements, "
                "travel advisories, current prices, upcoming events, exchange rates, recent reviews. "
                "Always use this for anything time-sensitive."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string", "description": "Specific search query."},
                    "max_results": {"type": "integer", "description": "Number of results (1-5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_destination_info",
            "description": "Get comprehensive travel info for a destination: highlights, best time to visit, climate, visa requirements. Searches the web for current info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "City or country name."},
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_destination_info_static",
            "description": "Get key facts about a specific destination: best time to visit, climate, highlights, language, currency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "Name of the destination, e.g. 'Tokyo', 'Bali', 'Paris'.",
                    }
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather and 5-day forecast for any city. Always call this when planning a trip or when the user asks about weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'Barcelona'."},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_itinerary",
            "description": "Generate a detailed day-by-day itinerary. Call this when the user asks for a travel plan or schedule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination":   {"type": "string", "description": "Destination city or country."},
                    "days":          {"type": "integer", "description": "Number of days."},
                    "travel_style":  {"type": "string", "description": "'relaxed', 'balanced', or 'packed'."},
                    "interests":     {"type": "string", "description": "Comma-separated interests, e.g. 'food, art, hiking'."},
                },
                "required": ["destination", "days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_budget",
            "description": "Search for current cost estimates for a trip. Use when the user asks about money, cost, or budget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination":   {"type": "string", "description": "Destination city."},
                    "days":          {"type": "integer", "description": "Number of days."},
                    "travel_style":  {"type": "string", "description": "'budget', 'mid-range', or 'luxury'."},
                },
                "required": ["destination", "days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_budget_static",
            "description": "Estimate the daily and total cost of a trip. Use this when the user asks about cost or budget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Destination city."},
                    "days":        {"type": "integer", "description": "Number of days."},
                    "travel_style": {
                        "type": "string",
                        "description": "Budget level: 'budget', 'mid-range', or 'luxury'.",
                    },
                },
                "required": ["destination", "days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_packing_list",
            "description": "Get a packing list tailored to the destination, trip length, and season.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Where the user is going."},
                    "days":        {"type": "integer", "description": "Length of the trip."},
                    "season": {
                        "type": "string",
                        "description": "Time of year: 'summer', 'winter', 'spring', or 'autumn'.",
                    },
                },
                "required": ["destination", "days", "season"],
            },
        },
    },
]


MODE = "normal"  # or "teaching" or "structured"

STRUCTURED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "travel_response",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "weather": {"type": "string"},
                "itinerary": {"type": "string"},
                "budget": {"type": "string"},
                "tips": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["summary"]
        }
    }
}

# =============================================================================
# PART 3: SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert travel agent with deep knowledge of destinations worldwide and access to real-time web search.
Your job is to help users plan amazing trips by giving specific, current, and genuinely useful travel advice.

Tool usage guide:
- User asks for destination ideas → call get_destinations (with a vibe if they mentioned one)
- User picks a destination → call get_destination_info + get_weather (run both) to get the facts, then share them naturally
- User wants an itinerary → call generate_itinerary, then write the full plan
- User asks about cost/budget → call estimate_budget
- User asks what to pack → call get_packing_list
- User asks about visas, events, current prices, advisories → search_web with a specific query
- Anything time-sensitive or that might have changed → search_web first

Rules:
- Always check weather when discussing a specific destination trip
- Be specific: name real restaurants, neighbourhoods, landmarks
- Be warm, enthusiastic, and specific. Don't just list places — paint a picture.
- If you don't have enough info (days? travel style? interests?), ask before calling tools
- After giving an itinerary, proactively offer: budget estimate, weather, packing tips
- Never make up visa requirements or prices — search for them"""

if MODE == "teaching":
    SYSTEM_PROMPT += """

When solving, use ReAct format:

Thought: what you are thinking
Action: which tool you will call
Observation: result of the tool
Repeat if needed.

Finally provide:
Final Answer: ...
"""

if MODE == "structured":
    SYSTEM_PROMPT += """

Return the final answer strictly in JSON format with keys:
- summary
- weather
- itinerary
- budget
- tips
"""

TEMPERATURE = 0.3

# =============================================================================
# PART 4: THE AGENT LOOP
# =============================================================================

def run_agent(user_message: str, history: list):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    log = ""
    for step in range(10):
        if MODE == "teaching":
            log += "🧠 Thinking...\n"
            yield log
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            response_format=STRUCTURED_SCHEMA if MODE == "structured" else None,
            temperature=TEMPERATURE,
        )

        msg           = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "stop":
            if MODE == "teaching":
                log += msg.content
                yield log
            elif msg.content:
                yield msg.content
            return

        if finish_reason == "tool_calls":
            if msg.content:
                if MODE == "teaching":
                    log += f"🧠 {msg.content}"
                    yield log
                else:
                    yield msg.content

            messages.append({
                "role":       "assistant",
                "content":    msg.content,
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                if MODE == "teaching":
                    log += f"⚙️ Action: {tool_name}({tool_args})"
                    yield log
                else:
                    yield f"⚙️ Using tool: {tool_name}"
                print(f"  → calling {tool_name}({tool_args})")
                if tool_name in TOOL_FUNCTIONS:
                    output = TOOL_FUNCTIONS[tool_name](**tool_args)
                else:
                    output = f"Error: unknown tool '{tool_name}'"
                if MODE == "teaching":
                    log += f"📦 Observation: {output[:300]}"
                    yield log
                else:
                    yield f"📦 Tool result received"
                print(f"  ← {output[:120]}...")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      output,
                })

    yield "Sorry, I couldn't finish planning your trip. Please try again."


# =============================================================================
# PART 5: GRADIO UI
# =============================================================================

def _to_openai_history(gradio_history: list) -> list:
    messages = []
    for entry in gradio_history:
        if isinstance(entry, dict):
            role    = entry.get("role", "")
            content = entry.get("content", "")
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            if entry[0]:
                messages.append({"role": "user",      "content": str(entry[0])})
            if entry[1]:
                messages.append({"role": "assistant", "content": str(entry[1])})
            continue
        else:
            continue
        if isinstance(content, list):
            text = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        else:
            text = str(content or "")
        if text.strip() and role in ("user", "assistant"):
            messages.append({"role": role, "content": text})
    return messages


def chat(user_message: str, gradio_history: list, mode: str, temperature: float):
    global MODE, TEMPERATURE
    MODE = mode
    TEMPERATURE = temperature
    history = _to_openai_history(gradio_history)
    yield from run_agent(user_message, history)


if __name__ == "__main__":
    mode_selector = gr.Radio(
        choices=["normal", "teaching", "structured"],
        value="normal",
        label="Mode"
    )
    temperature_slider = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.3,
            step=0.05,
            label="Temperature"
        )
    demo = gr.ChatInterface(
        fn=chat,
        title="Travel Agent",
        additional_inputs=[mode_selector, temperature_slider],
        description="Tell me where you want to go, how many days you have, and your travel style — I'll plan the whole trip.",
        examples=[
            ["I want a beach holiday for 7 days, budget is mid-range. Any suggestions?", "normal"],
            ["Plan me a 5-day itinerary for Tokyo, I like food and hidden gems.", "normal"],
            ["What's the best time of year to visit Bali?", "normal"],
            ["What's the weather like in Sweden right now? Is it a good time to visit?", "teaching"],
            ["How much would 10 days in Barcelona cost for a couple on a mid-range budget?", "normal"],
            ["What should I pack for a 2-week winter trip to Japan?", "normal"],
            ["What are the best food markets in Istanbul?", "structured"],
        ],
    )
    demo.launch(debug=True)