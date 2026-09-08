"""
Asset generator for modern 3D-styled SVG illustrations and avatars matching the reference UI.
"""
import os

assets_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(assets_dir, exist_ok=True)

# 1. 3D Golden Greek Temple / Academy (Course 1)
temple_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 160" width="100%" height="100%">
  <defs>
    <linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FDE047"/>
      <stop offset="50%" stop-color="#EAB308"/>
      <stop offset="100%" stop-color="#CA8A04"/>
    </linearGradient>
    <linearGradient id="goldDark" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#EAB308"/>
      <stop offset="100%" stop-color="#A16207"/>
    </linearGradient>
    <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#93C5FD"/>
      <stop offset="100%" stop-color="#BFDBFE"/>
    </linearGradient>
    <filter id="softShadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="#000" flood-opacity="0.15"/>
    </filter>
  </defs>
  <rect width="200" height="160" rx="16" fill="url(#skyGrad)"/>
  <g filter="url(#softShadow)" transform="translate(25, 20)">
    <!-- Roof Triangular Pediment -->
    <polygon points="75,10 15,40 135,40" fill="url(#goldGrad)"/>
    <polygon points="75,14 25,38 125,38" fill="url(#goldDark)" opacity="0.3"/>
    <!-- Crest Emblem -->
    <circle cx="75" cy="28" r="7" fill="#FEF08A"/>
    <path d="M72,28 L75,23 L78,28 Z" fill="#CA8A04"/>
    <!-- Entablature -->
    <rect x="18" y="40" width="114" height="10" rx="2" fill="url(#goldDark)"/>
    <!-- Columns -->
    <rect x="26" y="50" width="12" height="50" rx="3" fill="url(#goldGrad)"/>
    <rect x="52" y="50" width="12" height="50" rx="3" fill="url(#goldGrad)"/>
    <rect x="86" y="50" width="12" height="50" rx="3" fill="url(#goldGrad)"/>
    <rect x="112" y="50" width="12" height="50" rx="3" fill="url(#goldGrad)"/>
    <!-- Column details -->
    <line x1="32" y1="52" x2="32" y2="98" stroke="#CA8A04" stroke-width="1.5" opacity="0.5"/>
    <line x1="58" y1="52" x2="58" y2="98" stroke="#CA8A04" stroke-width="1.5" opacity="0.5"/>
    <line x1="92" y1="52" x2="92" y2="98" stroke="#CA8A04" stroke-width="1.5" opacity="0.5"/>
    <line x1="118" y1="52" x2="118" y2="98" stroke="#CA8A04" stroke-width="1.5" opacity="0.5"/>
    <!-- Base Steps -->
    <rect x="12" y="100" width="126" height="10" rx="3" fill="url(#goldDark)"/>
    <rect x="6" y="110" width="138" height="12" rx="4" fill="url(#goldGrad)"/>
  </g>
</svg>"""

# 2. 3D Character Celebrating (Course 2)
student_char_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 160" width="100%" height="100%">
  <defs>
    <linearGradient id="pinkGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FBCFE8"/>
      <stop offset="100%" stop-color="#F472B6"/>
    </linearGradient>
    <linearGradient id="skinGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#8D5524"/>
      <stop offset="100%" stop-color="#5C3317"/>
    </linearGradient>
    <filter id="charShadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="5" stdDeviation="5" flood-color="#000" flood-opacity="0.2"/>
    </filter>
  </defs>
  <rect width="200" height="160" rx="16" fill="url(#pinkGrad)"/>
  <g filter="url(#charShadow)" transform="translate(35, 10)">
    <!-- Raised Arms -->
    <path d="M15,50 Q10,20 28,15 Q35,25 25,55 Z" fill="url(#skinGrad)"/>
    <circle cx="28" cy="14" r="7" fill="url(#skinGrad)"/>
    <path d="M115,50 Q120,20 102,15 Q95,25 105,55 Z" fill="url(#skinGrad)"/>
    <circle cx="102" cy="14" r="7" fill="url(#skinGrad)"/>
    <!-- Body / Hoodie -->
    <path d="M35,65 Q65,55 95,65 L105,140 L25,140 Z" fill="#1E293B"/>
    <!-- Head -->
    <circle cx="65" cy="42" r="22" fill="url(#skinGrad)"/>
    <!-- Hair -->
    <path d="M43,38 Q65,15 87,38 Q78,20 65,22 Q50,20 43,38 Z" fill="#0F172A"/>
    <!-- Happy Eyes & Smile -->
    <circle cx="58" cy="40" r="3" fill="#FFF"/>
    <circle cx="59" cy="40" r="1.5" fill="#000"/>
    <circle cx="72" cy="40" r="3" fill="#FFF"/>
    <circle cx="73" cy="40" r="1.5" fill="#000"/>
    <path d="M57,49 Q65,58 73,49 Z" fill="#FFF"/>
    <!-- Celebration stars -->
    <path d="M10,25 L12,20 L17,22 L13,26 L15,31 L10,28 L5,31 L7,26 L3,22 L8,20 Z" fill="#FDE047" opacity="0.9"/>
    <path d="M120,30 L121,26 L125,27 L122,30 L124,34 L120,32 L116,34 L118,30 L115,27 L119,26 Z" fill="#FDE047" opacity="0.9"/>
  </g>
</svg>"""

# 3. 3D Gift Box (Sidebar bottom card)
gift_box_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 140" width="100%" height="100%">
  <defs>
    <linearGradient id="goldBox" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FCD34D"/>
      <stop offset="50%" stop-color="#F59E0B"/>
      <stop offset="100%" stop-color="#D97706"/>
    </linearGradient>
    <linearGradient id="greenBox" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#10B981"/>
      <stop offset="100%" stop-color="#047857"/>
    </linearGradient>
    <linearGradient id="redRibbon" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#EF4444"/>
      <stop offset="100%" stop-color="#B91C1C"/>
    </linearGradient>
    <filter id="giftShadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="8" stdDeviation="6" flood-color="#000" flood-opacity="0.15"/>
    </filter>
  </defs>
  <g filter="url(#giftShadow)" transform="translate(15, 10)">
    <!-- Box Body -->
    <rect x="20" y="55" width="90" height="65" rx="14" fill="url(#greenBox)"/>
    <!-- Lid -->
    <rect x="15" y="42" width="100" height="20" rx="8" fill="url(#goldBox)"/>
    <!-- Vertical Ribbon -->
    <rect x="56" y="42" width="18" height="78" rx="4" fill="url(#redRibbon)"/>
    <!-- Horizontal Ribbon -->
    <rect x="20" y="78" width="90" height="16" fill="url(#redRibbon)"/>
    <!-- Ribbon Bow Top -->
    <ellipse cx="48" cy="30" rx="16" ry="12" fill="url(#redRibbon)" transform="rotate(-25, 48, 30)"/>
    <ellipse cx="82" cy="30" rx="16" ry="12" fill="url(#redRibbon)" transform="rotate(25, 82, 30)"/>
    <circle cx="65" cy="35" r="8" fill="#DC2626"/>
  </g>
</svg>"""

# 4. Modern Avatars
def make_avatar(name, bg1, bg2, emoji):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100%" height="100%">
  <defs>
    <linearGradient id="grad_{name}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{bg1}"/>
      <stop offset="100%" stop-color="{bg2}"/>
    </linearGradient>
  </defs>
  <circle cx="50" cy="50" r="50" fill="url(#grad_{name})"/>
  <text x="50%" y="54%" font-size="44" text-anchor="middle" dominant-baseline="middle">{emoji}</text>
</svg>"""

with open(os.path.join(assets_dir, "course_temple.svg"), "w", encoding="utf-8") as f:
    f.write(temple_svg)

with open(os.path.join(assets_dir, "course_character.svg"), "w", encoding="utf-8") as f:
    f.write(student_char_svg)

with open(os.path.join(assets_dir, "gift_box.svg"), "w", encoding="utf-8") as f:
    f.write(gift_box_svg)

avatars = {
    "avatar_adam.svg": ("adam", "#38BDF8", "#0284C7", "👨‍🎓"),
    "avatar_michael.svg": ("michael", "#F43F5E", "#E11D48", "🧑‍💻"),
    "avatar_sophia.svg": ("sophia", "#A855F7", "#7C3AED", "👩‍🔬"),
    "avatar_emily.svg": ("emily", "#10B981", "#059669", "👩‍🎓"),
    "avatar_instructor1.svg": ("inst1", "#F59E0B", "#D97706", "👨‍🏫"),
    "avatar_instructor2.svg": ("inst2", "#EC4899", "#DB2777", "👩‍🏫")
}

for filename, (name, bg1, bg2, emoji) in avatars.items():
    with open(os.path.join(assets_dir, filename), "w", encoding="utf-8") as f:
        f.write(make_avatar(name, bg1, bg2, emoji))

print("Assets generated successfully!")
