#!/usr/bin/env python3
"""
DefectBot Training Dataset Generator
Generates 60+ synthetic training samples covering building and mining defects.
Each sample: instruction (technician report) + response (follow-up questions + diagnosis + manager summary).

Run:  python data/make_dataset.py
Output: data/train.jsonl
"""

import json
import os

# Output path relative to project root
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "train.jsonl")

# ---------------------------------------------------------------------------
# BUILDING DEFECTS — 30+ samples
# ---------------------------------------------------------------------------
BUILDING_SAMPLES = [

    # ── AC Condensation ────────────────────────────────────────────────────
    {
        "instruction": "I see water dripping from the ceiling in the server room near the AC unit. The stain is directly below the air handler.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the water dripping only when the AC is running, or does it continue even when the unit is off?
2. Is the ceiling tile discolored yellow-brown (mineral staining from pipe leak) or white/clear (fresh condensation)?
3. What is the ambient humidity level in the server room, and is the AC drip pan accessible for inspection?

DIAGNOSIS:
Root cause: AC condensate drain line blockage causing overflow from the drip pan. The location directly below the air handler strongly indicates condensation overflow rather than a pipe leak above.
Contributing factors: Biological growth (algae/mould) in the condensate drain line, kinked or improperly sloped drain pipe, or high humidity load exceeding the AC unit's drainage capacity.
Distinguishing from pipe leak: Pipe leak water typically has a yellow-brown mineral stain and continues when AC is off; condensation water is clear and stops when AC is turned off.

MANAGER SUMMARY
Severity: HIGH
Issue: Water infiltration in server room ceiling from AC condensate overflow — active risk to IT equipment.
Root Cause: Blocked condensate drain line in air handling unit causing pan overflow.
Immediate Action: Turn off AC unit, place drip trays under all IT equipment, engage HVAC technician to clear drain line and inspect drip pan within 4 hours.
Recommended Timeline: Emergency inspection today; drain cleaning within 24 hours; full HVAC service within 1 week.
Estimated Cost Impact: $300–$800 drain service; potential $5,000–$50,000+ if server equipment is damaged."""
    },
    {
        "instruction": "Water stain on ceiling in corridor, brownish, roughly 60cm wide. It's on level 3 and there's a bathroom on level 4 directly above.",
        "response": """FOLLOW-UP QUESTIONS:
1. Does the stain grow larger after heavy rain, or does it appear linked to bathroom usage patterns (mornings, post-lunch)?
2. Is there any visible cracking in the stain area, or is the plaster soft/bubbling to the touch?
3. Has the bathroom above recently had any plumbing work done, or has this stain appeared after a specific event?

DIAGNOSIS:
Root cause: Bathroom pipe leak from level 4 — most likely a waste pipe connection, floor drain seal, or toilet wax ring failure. The brown staining indicates water carrying dissolved materials (sewage or rust), consistent with a waste or supply line above.
Contributing factors: Corroded pipe joints, failed waterproofing membrane under bathroom tiles, or cracked floor drain assembly allowing wastewater to seep into the slab.
Risk escalation: If sewage water is involved, there is a hygiene/health risk; affected ceiling materials must be treated as contaminated.

MANAGER SUMMARY
Severity: HIGH
Issue: Active ceiling leak in Level 3 corridor — brownish staining indicates potential sewage or corroded supply pipe leak from Level 4 bathroom.
Root Cause: Probable failure of bathroom plumbing connection or waterproofing membrane at Level 4.
Immediate Action: Restrict access to affected corridor section, isolate Level 4 bathroom water supply, engage licensed plumber for CCTV pipe inspection within 24 hours.
Recommended Timeline: Plumbing inspection within 24 hours; repair within 3–5 days; ceiling remediation 1–2 weeks after repair.
Estimated Cost Impact: $1,500–$4,000 plumbing repair; $2,000–$6,000 ceiling remediation and repainting."""
    },
    {
        "instruction": "The ceiling in the ground floor car park is dripping water during heavy rain. There are multiple drip points across a 10-metre span.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is there a slab above the carpark (occupied floor or podium) or is it directly exposed to the roof?
2. Are the drip points concentrated near expansion joints, column bases, or randomly distributed?
3. How long has this been occurring — first wet season or has it been ongoing for multiple years?

DIAGNOSIS:
Root cause: Failed waterproofing membrane on the podium slab or roof deck above. Multiple distributed drip points over a large span indicate widespread membrane failure rather than a localized pipe leak.
Contributing factors: UV degradation of the membrane, failed expansion joint sealant, cracked slab allowing water tracking, or clogged roof drains causing ponding that overwhelms membrane edges.
Risk: Ongoing water infiltration causes steel rebar corrosion (spalling risk), concrete carbonation, and potential structural weakening over time.

MANAGER SUMMARY
Severity: HIGH
Issue: Multiple active water infiltration points in carpark ceiling over 10m span during rain — indicates widespread waterproofing failure.
Root Cause: Degraded or failed waterproofing membrane on podium slab/roof deck above carpark.
Immediate Action: Deploy drip trays, barricade affected parking bays, engage waterproofing specialist for membrane inspection within 48 hours.
Recommended Timeline: Specialist inspection within 48 hours; temporary sealant patch within 1 week; full membrane remediation planned for dry season.
Estimated Cost Impact: $15,000–$80,000+ depending on extent of membrane replacement required."""
    },

    # ── Structural Cracks ──────────────────────────────────────────────────
    {
        "instruction": "There's a diagonal crack running from the corner of a window down to the floor. It's about 3mm wide and the wall is brick.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the crack wider at the top (at window corner) or at the bottom, and is there a step pattern following the mortar joints?
2. Has there been any recent excavation, tree removal, or construction activity within 10 metres of this wall?
3. When did you first notice the crack — after a wet/dry season cycle, or following a specific event?

DIAGNOSIS:
Root cause: Differential foundation settlement causing the masonry wall to crack in tension at the weakest point — the window corner stress concentration. Diagonal cracks from window corners following mortar bed joints are a classic sign of subsidence or differential settlement.
Contributing factors: Clay soil shrinkage in dry season (soil movement), nearby tree root activity desiccating the subsoil, or inadequate footing depth. A 3mm crack width indicates this is beyond cosmetic — it is a structural concern.
Monitoring needed: Without crack monitoring, it is impossible to determine if movement is ongoing or historic.

MANAGER SUMMARY
Severity: HIGH
Issue: 3mm diagonal structural crack from window corner in brick wall — indicative of differential foundation settlement.
Root Cause: Probable foundation subsidence due to soil movement, tree root activity, or inadequate footing bearing.
Immediate Action: Install crack monitoring tell-tales immediately; commission geotechnical and structural engineer inspection within 48 hours; check for soil movement or nearby tree roots.
Recommended Timeline: Engineer inspection within 48 hours; monitoring period 4–8 weeks; remediation based on engineer report.
Estimated Cost Impact: $800–$2,000 engineering assessment; $10,000–$100,000+ for underpinning or remediation depending on severity."""
    },
    {
        "instruction": "Hairline cracks on the external render, multiple cracks running vertically and horizontally, looks like a map pattern.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the render cement-based or acrylic, and how old is it approximately?
2. Do the cracks penetrate through to the masonry behind, or are they only on the surface render layer?
3. Is the cracking concentrated on south-facing or shaded walls, or uniform across all elevations?

DIAGNOSIS:
Root cause: Map cracking (crazing) in cement render due to shrinkage during curing or thermal expansion/contraction cycling. This pattern is characteristic of render that was applied too thick, mixed incorrectly, or cured too quickly in hot/dry conditions.
Contributing factors: Differential thermal movement between the render and substrate, moisture absorption causing swelling followed by drying shrinkage, or carbonation of the cement binder over time.
Risk level: If cracks do not penetrate the substrate and are <0.3mm wide, they are primarily cosmetic. Wider cracks allow moisture ingress leading to efflorescence and eventual render detachment.

MANAGER SUMMARY
Severity: MEDIUM
Issue: Map cracking pattern across external render — primarily cosmetic but creates moisture ingress pathways if untreated.
Root Cause: Render shrinkage cracking from improper cure, over-thick application, or long-term thermal cycling.
Immediate Action: Probe affected areas for hollow/delaminated render (tap test); seal any cracks wider than 0.3mm with flexible filler to prevent water ingress.
Recommended Timeline: Inspection and tap testing within 2 weeks; crack sealing within 1 month; full render repair/replacement during next planned maintenance cycle.
Estimated Cost Impact: $2,000–$8,000 for localized repairs; $15,000–$40,000 for full re-render of affected elevations."""
    },
    {
        "instruction": "Large horizontal crack running along the mortar joint between Level 1 and Level 2 external brick wall, approximately 8 metres long.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the crack open (gap visible) or is it a shear crack with one course of brick displaced relative to the other?
2. Is there any outward bowing or bulging of the brickwork above or below the crack when viewed from a distance?
3. When was the building constructed and has there been any recent loading change — added plant equipment on the roof, or removal of internal walls?

DIAGNOSIS:
Root cause: Horizontal crack in mortar bed joint at this location is a serious structural warning sign. It indicates either: (a) compressive overloading causing mortar crushing, (b) lateral earth pressure from retained soil, (c) lintel failure causing load redistribution, or (d) thermal/moisture movement of the outer leaf in a cavity wall.
Critical concern: If accompanied by wall bowing, this represents imminent risk of wall collapse and requires immediate action.

MANAGER SUMMARY
Severity: CRITICAL
Issue: 8-metre horizontal crack in external brick mortar joint between Level 1–2 — potential structural wall failure risk.
Root Cause: Suspected compressive overloading, lateral pressure, or outer leaf lateral instability — requires urgent structural engineering assessment.
Immediate Action: IMMEDIATELY cordon off the area below the affected wall (minimum 2-metre exclusion zone), notify building owner, engage structural engineer for emergency inspection within 4 hours. Do not occupy rooms adjacent to this wall until cleared.
Recommended Timeline: Emergency structural inspection today; shoring/propping if required within 24 hours; repair design within 1 week.
Estimated Cost Impact: $3,000–$5,000 emergency engineering; $20,000–$200,000+ depending on extent of structural repair."""
    },

    # ── Spalling ──────────────────────────────────────────────────────────
    {
        "instruction": "Chunks of concrete falling from the carpark soffit. I can see rusty steel poking through in some places.",
        "response": """FOLLOW-UP QUESTIONS:
1. What is the diameter of the exposed rebar — approximately pencil-thin (10mm) or finger-thick (16mm+)?
2. How large is the spalled area — a single patch or multiple patches across the ceiling?
3. Is this a suspended slab (with traffic above) or a ground-level soffit?

DIAGNOSIS:
Root cause: Chloride-induced or carbonation-induced rebar corrosion causing expansive rust formation that fractures the concrete cover. In carparks, chloride ingress from de-icing salts or coastal air is the primary cause. The rust products (iron oxides) expand to 6–10× the volume of the original steel, generating internal tensile forces that crack and spall the concrete cover.
Risk: Falling concrete chunks are an immediate life-safety hazard. Exposed rebar with active corrosion will continue to accelerate without treatment, and structural capacity is being progressively reduced.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Active concrete spalling with exposed corroded rebar in carpark soffit — immediate falling object hazard and structural degradation.
Root Cause: Rebar corrosion (chloride or carbonation induced) causing expansive cracking and concrete cover loss.
Immediate Action: CLOSE the affected bay immediately. Install safety netting or hoarding below. Engage structural engineer and concrete repair specialist within 24 hours. Map all spalled and cracked areas — tap test the surrounding soffit for delamination.
Recommended Timeline: Emergency make-safe within 24 hours; full structural assessment within 1 week; concrete repair works within 4–8 weeks.
Estimated Cost Impact: $500–$2,000 emergency make-safe; $15,000–$100,000+ for structural concrete repair and cathodic protection."""
    },
    {
        "instruction": "Concrete spalling on external column at ground floor, base of column. Surface is flaking off, no visible rebar yet.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the column base at grade level or slightly below, and is there visible water pooling or moisture around the column base?
2. What is the approximate extent of spalling — localized patch under 20cm, or wrapping around the column circumference?
3. Is the column in a coastal location or a high-chloride environment (carpark, industrial area, swimming pool)?

DIAGNOSIS:
Root cause: Moisture-driven concrete degradation at the column base — likely freeze-thaw cycling damage (in cool climates) or alkali-silica reaction (ASR), or early-stage chloride ingress from ground splash. The column base is a common failure zone due to persistent moisture exposure from ground splash and poor drainage.
If rebar is not yet exposed, this is at an early-moderate stage where repair is still straightforward.
Risk progression: Without repair, moisture continues penetrating, rebar corrodes, and the column base loses cross-section — a structural risk to the building.

MANAGER SUMMARY
Severity: MEDIUM
Issue: Concrete spalling at ground floor column base — early-stage deterioration without rebar exposure, but requires prompt intervention to prevent escalation.
Root Cause: Moisture ingress at column base causing concrete disintegration (freeze-thaw, carbonation, or early chloride attack).
Immediate Action: Mark affected columns for monitoring; engage structural engineer to assess extent and specify repair method within 2 weeks.
Recommended Timeline: Engineering assessment within 2 weeks; concrete patch repair within 6 weeks; apply protective coating post-repair.
Estimated Cost Impact: $500–$2,000 per column for repair and coating."""
    },

    # ── Efflorescence ──────────────────────────────────────────────────────
    {
        "instruction": "White powdery deposits on the external brick wall, looks like salt crystals. It's near the base of the wall.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the white deposit soft and powdery (can be brushed off) or hard and crystalline (difficult to remove)?
2. Does the deposit appear and grow worse during or after rain, then disappear in dry weather?
3. Is the wall a cavity wall or solid masonry, and is there any sign of damp on the internal face?

DIAGNOSIS:
Root cause: Efflorescence — soluble salts (calcium carbonate, sodium sulfate) being transported to the surface by moisture movement through the masonry. The source is water infiltrating through the wall face, rising damp from below, or mortar salts. Near the base suggests rising damp contribution.
Contributing factors: Inadequate damp-proof course (DPC), porous mortar, or lack of weep holes in cavity wall allowing water to accumulate and drive salts to the surface.
Note: Efflorescence itself does not cause structural damage, but it is a symptom of moisture movement that can cause long-term masonry deterioration and internal damp.

MANAGER SUMMARY
Severity: LOW
Issue: Efflorescence (salt deposits) on external brick wall base — cosmetic issue but symptomatic of moisture movement that requires moisture source investigation.
Root Cause: Soluble salt migration driven by water movement through masonry — possible rising damp or inadequate DPC.
Immediate Action: Brush off loose deposits (dry method); investigate for rising damp — check DPC continuity and external ground levels.
Recommended Timeline: Moisture investigation within 1 month; DPC repair or injection if required within 3 months; clean and seal brickwork during next planned maintenance.
Estimated Cost Impact: $500–$2,000 for DPC investigation and treatment; $1,000–$3,000 for brickwork cleaning and sealing."""
    },

    # ── Mould ─────────────────────────────────────────────────────────────
    {
        "instruction": "Black mould on the bathroom ceiling and top of the walls. It's in multiple bathrooms on the same floor.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the exhaust fan in these bathrooms functioning — can you feel airflow when it runs?
2. Do the mould patches appear at the wall-ceiling junction specifically, or uniformly across the ceiling?
3. Are the affected bathrooms on an upper floor below the roof, or mid-building, and has the building recently been refurbished (new insulation)?

DIAGNOSIS:
Root cause: Inadequate ventilation causing high humidity and condensation on cold ceiling surfaces, promoting mould (Cladosporium or Stachybotrys) growth. Multiple bathrooms on the same floor indicates a systemic ventilation failure — likely a blocked exhaust duct riser, failed exhaust fans, or inadequate fresh air supply.
Health risk: Black mould releases spores that cause respiratory issues. This is a health and safety concern for occupants, particularly the immunocompromised.
Contributing factors: Exhaust fan connected to a common duct riser that is blocked, insufficient fan capacity, or bathroom not being ventilated during/after use.

MANAGER SUMMARY
Severity: HIGH
Issue: Black mould in multiple bathrooms on same floor — health hazard indicating systemic ventilation failure.
Root Cause: Inadequate exhaust ventilation causing condensation and mould growth on ceiling and wall surfaces.
Immediate Action: Increase ventilation — open windows where possible; inspect and test all exhaust fans; check common duct riser for blockage. If occupants have respiratory conditions, consider temporary relocation.
Recommended Timeline: Exhaust fan inspection within 24 hours; mould remediation by licensed hygienist within 1 week; ventilation upgrade within 4 weeks.
Estimated Cost Impact: $2,000–$5,000 mould remediation; $3,000–$8,000 ventilation upgrade depending on extent."""
    },
    {
        "instruction": "Green mould growing on the bottom of an external wall inside the apartment. Only on that one wall, near the floor.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the affected wall an external wall facing a particularly shaded or south-facing direction?
2. Is there a garden bed, paving, or soil level outside that is at or above the internal floor level?
3. Is there a damp-proof course (DPC) visible in the external brickwork, or has any recent landscaping raised the external ground level?

DIAGNOSIS:
Root cause: Rising damp — groundwater wicking up through the masonry by capillary action, carrying moisture into the internal wall surface. Rising damp characteristically affects the lower 1–1.5m of walls and is most pronounced on external walls. The mould is a secondary effect of persistent moisture.
Contributing factors: Absent, failed, or bridged damp-proof course; external ground level raised above DPC level by landscaping; blocked weep holes.

MANAGER SUMMARY
Severity: MEDIUM
Issue: Rising damp causing mould growth on internal base of external wall — ongoing moisture ingress damaging finishes and creating mould risk.
Root Cause: Rising damp due to failed or absent DPC, or external ground level bridging the DPC.
Immediate Action: Check external ground level relative to internal floor level; inspect for DPC presence and continuity. Remove affected internal plaster and treat with mould-resistant primer.
Recommended Timeline: Investigation within 2 weeks; DPC injection or external membrane treatment within 6 weeks; internal replastering with waterproof render within 8 weeks.
Estimated Cost Impact: $3,000–$8,000 for DPC treatment and internal repair."""
    },

    # ── Paint Deterioration ────────────────────────────────────────────────
    {
        "instruction": "Paint on external wall is bubbling and peeling off in large sheets. The wall faces west and gets afternoon sun.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the substrate under the peeling paint brick, concrete, or fibre cement sheeting?
2. When was the wall last painted, and was the surface primed before painting?
3. Are there signs of moisture on the internal side of this wall — any dampness, staining, or condensation on the inside face?

DIAGNOSIS:
Root cause: Paint delamination due to moisture vapour pressure from the substrate — moisture trapped behind the paint film creates pressure that lifts the paint. West-facing walls experience intense afternoon solar heating that drives moisture vapour outward through the wall, blistering the paint film from behind.
Contributing factors: Paint applied over damp substrate, inadequate primer, or incompatible paint system (e.g., oil-based paint over a breathable substrate). Osmotic blistering if water-borne topcoat over oil-based primer.

MANAGER SUMMARY
Severity: LOW
Issue: Paint bubbling and peeling on west-facing external wall — cosmetic deterioration driven by moisture vapour pressure and solar heat.
Root Cause: Moisture vapour escaping through wall blistering paint from substrate, exacerbated by afternoon solar heat gain.
Immediate Action: Probe wall moisture content; investigate and resolve any moisture source before repainting.
Recommended Timeline: Moisture investigation within 2 weeks; full strip, prime, and repaint in cooler weather (avoid direct sun) within 2–3 months.
Estimated Cost Impact: $3,000–$10,000 depending on wall area for preparation and repainting."""
    },

    # ── Roof Leakage ──────────────────────────────────────────────────────
    {
        "instruction": "Water coming through the ceiling light fitting in the top floor office during heavy rain.",
        "response": """FOLLOW-UP QUESTIONS:
1. Does the leak occur in all heavy rain events or only during wind-driven rain from a specific direction?
2. What type of roof is above — flat (membrane/bitumen), metal deck, or tiled pitched roof?
3. Are there any roof penetrations (AC units, vents, pipes) near the affected ceiling area?

DIAGNOSIS:
Root cause: Roof membrane breach or failed flashing at a penetration point directly above. Water tracking to the light fitting indicates the entry point is offset from the leak location — water travels along the structural deck until it finds an exit path (the light fitting opening). Entry points commonly include: failed AC unit curb flashing, cracked membrane at drain upturns, or deteriorated penetration seals.
Safety note: Water in electrical fittings creates electrocution and fire risk — the electrical circuit should be isolated immediately.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Active roof leak through ceiling light fitting during rain — immediate electrical hazard.
Root Cause: Roof membrane or penetration flashing failure allowing rainwater to track to ceiling light fitting.
Immediate Action: IMMEDIATELY isolate the electrical circuit to the affected light fitting. Deploy drip collection. Engage roofing contractor for emergency inspection within 24 hours. Do not restore power to circuit until water damage assessed by electrician.
Recommended Timeline: Emergency roof inspection and temporary repair within 24 hours; permanent flashing/membrane repair within 1 week; electrical inspection and reinstatement within 48 hours.
Estimated Cost Impact: $500–$2,000 emergency roof patch; $2,000–$8,000 permanent repair; $500–$1,500 electrical assessment and repair."""
    },
    {
        "instruction": "Tiled roof, I can see a cracked tile from the attic. There's a small wet patch on the timber rafter below it.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the cracked tile at the ridge (top), in the field (middle), or at the eave (bottom) of the roof?
2. Is the wet patch on the rafter isolated to directly below the cracked tile, or is there a broader wet area?
3. Is there sarking (foil or paper underlay) under the tiles, and is it intact below the crack?

DIAGNOSIS:
Root cause: Cracked roof tile allowing direct rainwater entry onto the timber rafter. Tiles crack from thermal cycling fatigue, foot traffic damage, hail impact, or UV embrittlement. If sarking is present but compromised, water bypasses the secondary waterproofing line.
Risk: Wet timber rafters will decay over time; prolonged moisture exposure causes mould and structural timber weakening. However, a single cracked tile is a straightforward repair if addressed promptly.

MANAGER SUMMARY
Severity: MEDIUM
Issue: Cracked roof tile causing localised rafter moisture — requires prompt tile replacement to prevent timber decay.
Root Cause: Mechanically or thermally cracked tile creating a direct water ingress point.
Immediate Action: Temporarily place a waterproof patch (roofing tape or lead flashing) over the cracked tile if safe to access; dry out attic space.
Recommended Timeline: Permanent tile replacement within 1–2 weeks; inspect surrounding tiles for hairline cracks; check rafter moisture content and dry out.
Estimated Cost Impact: $200–$600 for tile replacement; $500–$2,000 if sarking also requires repair."""
    },

    # ── Rising Damp ───────────────────────────────────────────────────────
    {
        "instruction": "Damp patches on internal ground floor walls, they're about 600mm high, tide mark staining, and the plaster is crumbling.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the damp present on all ground floor walls uniformly, or only on walls adjacent to external ground?
2. Is there a basement, subfloor void, or direct slab-on-ground construction?
3. Has there been any recent heavy rainfall or change in external drainage around the building?

DIAGNOSIS:
Root cause: Classic rising damp from capillary rise of groundwater through the masonry wall. The tide mark (horizontal stain line) at approximately 600mm height is pathognomonic of rising damp — the moisture rises until the evaporation rate equals the capillary rise rate. Crumbling plaster is caused by hygroscopic salt crystallisation as moisture evaporates within the plaster matrix, expanding and shattering it.
Root cause confirmed by: Uniform height tide mark, salt efflorescence, plaster deterioration — all consistent with rising damp rather than penetrating damp (which would show vertical staining from above).

MANAGER SUMMARY
Severity: MEDIUM
Issue: Active rising damp on multiple ground floor walls — tide marks at 600mm height with plaster failure indicate established capillary rise.
Root Cause: Failed or absent damp-proof course (DPC) allowing groundwater to wick up through masonry walls.
Immediate Action: Confirm DPC presence and condition (expose base of external masonry); check external ground level — if above DPC, regrade soil away from building.
Recommended Timeline: DPC investigation within 2 weeks; chemical DPC injection treatment within 4–8 weeks; allow wall to dry (minimum 1 month per 25mm wall thickness) before replastering with waterproof render.
Estimated Cost Impact: $3,000–$10,000 for DPC injection and internal replastering; $1,000–$3,000 additional for drainage improvement."""
    },

    # ── Additional Building Samples ────────────────────────────────────────
    {
        "instruction": "I can smell sewage in the basement carpark even when no one is using the toilets.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the smell constant or does it intensify at certain times of day or after rainfall?
2. Are there floor drains in the basement — when did they last have water poured into them?
3. Has any recent plumbing work been done to the sewer lines in the building?

DIAGNOSIS:
Root cause: Dry floor drain traps allowing sewer gases to escape into the basement. Floor drain P-traps require periodic water topping to maintain the water seal that blocks sewer gas. In infrequently used areas like basements, the trap water evaporates over weeks.
Alternative cause: Cracked sewer pipe beneath the slab, allowing gases to permeate through concrete into the space.
Sewer gas contains hydrogen sulphide (H₂S) — toxic and flammable at high concentrations.

MANAGER SUMMARY
Severity: HIGH
Issue: Sewer gas odour in basement carpark — potential health and fire hazard from hydrogen sulphide gas.
Root Cause: Likely evaporated floor drain P-traps allowing sewer gas backflow; secondary possibility of cracked underground sewer pipe.
Immediate Action: Pour water into all floor drains immediately (prime the traps); increase ventilation; if smell persists, commission plumber for CCTV sewer inspection.
Recommended Timeline: Trap priming today; if odour persists after 24 hours, plumber inspection within 48 hours.
Estimated Cost Impact: $0 if trap priming resolves; $2,000–$8,000 if sewer repair required."""
    },
    {
        "instruction": "Glass in the fire-rated door on Level 2 stairwell has a crack running from corner to corner.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is this a wire glass (with embedded wire mesh), ceramic glass (no wire, smooth), or standard float glass?
2. Does the crack appear to follow a straight line (thermal/mechanical stress) or is it branching (impact)?
3. Is the door in a high-traffic area where impact from a trolley or door slam could have caused it?

DIAGNOSIS:
Root cause: Mechanical impact crack in fire-rated door glazing. Fire-rated door glass is either wired glass or ceramic glass specifically rated to maintain integrity under heat — a cracked pane fails its fire rating and must be replaced to maintain compartmentation.
Regulatory issue: A cracked fire door glass is a non-compliance with fire safety regulations. The door's fire rating is compromised until the glass is replaced.

MANAGER SUMMARY
Severity: HIGH
Issue: Cracked glass in fire-rated stairwell door — fire safety non-compliance requiring immediate remediation.
Root Cause: Mechanical impact causing crack across fire-rated glazing, compromising fire compartmentation integrity.
Immediate Action: Tag door as non-compliant; contact fire door specialist for emergency glass replacement within 24–48 hours; do not delay as this affects building fire safety compliance.
Recommended Timeline: Replacement within 48 hours; update fire door inspection register.
Estimated Cost Impact: $400–$1,500 for fire-rated glass replacement depending on pane size."""
    },
    {
        "instruction": "Expansion joint sealant between two building wings has split open and I can see daylight through it on Level 5.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the expansion joint horizontal (floor/slab joint) or vertical (wall joint), and approximately how wide is the gap?
2. Is there water staining or active water ingress near the joint?
3. How old is the sealant — is it a brittle failed rubber or a clean new sealant that has torn at the bond line?

DIAGNOSIS:
Root cause: Failed expansion joint sealant — either exceeded its design movement range, reached end of service life (sealants typically last 10–15 years), or was applied to an inadequately prepared surface causing adhesive failure. Expansion joints are critical waterproofing elements between building sections that move independently.
Risk: Open expansion joint on upper floors is a direct rainwater entry path that can cause extensive internal water damage to multiple floors below.

MANAGER SUMMARY
Severity: HIGH
Issue: Open expansion joint on Level 5 — direct water ingress pathway across multiple floors.
Root Cause: Failed building expansion joint sealant due to age, movement exceedance, or adhesion failure.
Immediate Action: Temporary weatherproofing (tape or temporary flexible sealant) immediately if rain is forecast; engage building envelope specialist for assessment within 1 week.
Recommended Timeline: Temporary seal within 24 hours if rain risk; permanent joint reseal within 4 weeks using appropriate movement-rated sealant and bond-breaker tape.
Estimated Cost Impact: $1,500–$5,000 per joint for professional reseal."""
    },
    {
        "instruction": "Basement waterproof membrane has a visible blister/bubble about 30cm across on the floor.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the membrane a sheet-applied type (torch-on bitumen, HDPE) or a liquid-applied coating?
2. When you press on the blister, does it feel firm (trapped gas) or does water come out if punctured?
3. Is the basement below the water table, and is there any hydrostatic pressure reported?

DIAGNOSIS:
Root cause: Hydrostatic uplift blister — water under positive pressure between the waterproofing membrane and the substrate is lifting the membrane. This occurs when groundwater pressure exceeds the bond strength of the membrane to the slab. This is a serious failure indicating the membrane is subject to conditions beyond its design.
Risk: If the blister ruptures, water will flood into the basement. Additional blistering may occur across the floor.

MANAGER SUMMARY
Severity: HIGH
Issue: Waterproofing membrane blister on basement floor — hydrostatic uplift indicating active groundwater pressure beneath membrane.
Root Cause: Groundwater hydrostatic pressure exceeding membrane bond strength, lifting and delaminating the membrane.
Immediate Action: Do not puncture the blister; monitor size daily; engage waterproofing engineer to assess groundwater levels and recommend drainage relief or membrane replacement strategy.
Recommended Timeline: Engineering assessment within 1 week; dewatering strategy or membrane replacement within 4–8 weeks.
Estimated Cost Impact: $5,000–$30,000+ depending on basement area and groundwater management approach."""
    },
    {
        "instruction": "Stairwell handrail is loose — the balustrade post wobbles significantly when pressure is applied. It's on the main entry staircase.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the post bolted through the stair structure (bolt-through fixing) or surface-mounted with a base plate?
2. Is the wobble at the base of the post (fixing failure) or along the rail itself (rail-to-post connection failure)?
3. How many posts are affected, and is there visible corrosion at the base plate or fixing bolts?

DIAGNOSIS:
Root cause: Loose balustrade post from failed base plate fixing — either corroded bolts have lost clamping force, the base plate has cracked, or the substrate (concrete or timber) around the fixing has deteriorated. A loose handrail post is a direct fall hazard and fails building code requirements for balustrading (must withstand 0.75–1 kN horizontal load).
Risk: If a person falls against the balustrade, the loose post could fail, causing a fall from height.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Loose handrail balustrade post on main entry staircase — fall hazard and building code non-compliance.
Root Cause: Failed base plate fixing (corroded bolts or substrate deterioration) reducing post lateral load resistance to near zero.
Immediate Action: TAPE OFF the affected stair section. Install temporary rope barrier. Engage qualified tradesperson (structural or metalwork) to inspect and re-fix or replace base plate within 24 hours.
Recommended Timeline: Make-safe today; permanent repair within 48 hours; inspect all other balustrade posts within 1 week.
Estimated Cost Impact: $300–$1,500 per post repair; $2,000–$8,000 if multiple posts require replacement."""
    },
    {
        "instruction": "I can see a large damp patch on the ceiling of the lobby — it appeared after the heavy rain last night. The patch is roughly 1.5 metres across.",
        "response": """FOLLOW-UP QUESTIONS:
1. What is directly above the lobby ceiling — a roof, another floor, or a plant/mechanical room?
2. Is the damp patch growing since it first appeared, or has it stabilised?
3. Are there any ceiling tiles softening or bowing under the weight of the water?

DIAGNOSIS:
Root cause: Rainwater infiltration through the roof or upper floor causing ceiling saturation. A 1.5-metre patch appearing after rain strongly implicates the roof. In lobbies, common failure points include: roof drain overflow or blockage causing ponding, failed flashing at parapet walls, or HVAC roof penetrations with failed sealant.
Risk: Saturated ceiling tiles can collapse without warning, posing a falling object hazard to lobby occupants. Electrical risk if lighting is embedded in the wet ceiling area.

MANAGER SUMMARY
Severity: HIGH
Issue: Active rain-induced ceiling leak in building lobby — falling ceiling tile risk and potential electrical hazard.
Root Cause: Roof water infiltration reaching lobby ceiling — likely blocked drain, failed flashing, or roof penetration seal failure.
Immediate Action: Barricade the area below the wet ceiling; check for and isolate any electrical fixtures in the wet zone; deploy drip collection; engage roofer for emergency inspection.
Recommended Timeline: Emergency roof inspection within 24 hours; temporary repair within 24–48 hours; permanent repair within 1–2 weeks.
Estimated Cost Impact: $500–$2,000 emergency inspection and patch; $3,000–$10,000 permanent repair."""
    },
    {
        "instruction": "Fire damper inspection revealed a failed damper in the HVAC duct on Level 4 — the blade won't close when tested.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the damper a fusible link type (melts at 72°C to close) or an electrically actuated type (closes on fire signal)?
2. Is the blade physically stuck (debris, corrosion) or is the actuator/mechanism failing to drive it?
3. Has this damper been accessible for regular inspection, or has it been behind sealed access panels?

DIAGNOSIS:
Root cause: Fire damper blade failure — either mechanism seizure from corrosion/debris accumulation or actuator failure. Fire dampers must close within 30 seconds of heat or signal activation to prevent fire and smoke spread through ductwork. A failed damper is a fire safety non-compliance that must be rectified under AS1682 (or relevant jurisdiction standard).
Regulatory: Failed fire dampers must be reported and rectified — the building's fire safety statement/certificate may be affected.

MANAGER SUMMARY
Severity: HIGH
Issue: Failed fire damper on Level 4 HVAC duct — fire safety non-compliance requiring urgent rectification.
Root Cause: Damper blade or actuator failure preventing closure — mechanism seizure or actuator fault.
Immediate Action: Tag the damper as non-compliant; notify building fire safety officer; engage licensed fire protection contractor for repair within 48 hours.
Recommended Timeline: Repair within 48 hours; re-test after repair; update fire safety register; schedule annual re-inspection of all fire dampers.
Estimated Cost Impact: $500–$2,000 per damper repair/replacement; $2,000–$5,000 for full fire damper inspection program."""
    },
    {
        "instruction": "Timber deck boards are soft and spongy in one corner, a couple of boards have actually snapped when walked on.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the deck above a habitable room or over an open carpark/ground?
2. Are the deck boards treated timber, hardwood, or composite? How old is the deck?
3. Is there a waterproof membrane or flashing under the deck boards where they meet the building wall?

DIAGNOSIS:
Root cause: Advanced timber decay from prolonged moisture exposure. Soft and collapsing boards indicate the wood fibres have been destroyed by fungal (brown or white rot) activity. The corner location is typical — corners trap debris, hold moisture, and the junction with the building wall often has flashing failures that keep the area perpetually damp.
Safety risk: Collapsed deck boards present a fall risk. The structural framing below the decking may also be compromised.

MANAGER SUMMARY
Severity: HIGH
Issue: Collapsed and soft deck boards in corner — safety fall hazard and advanced timber decay requiring immediate isolation.
Root Cause: Fungal timber decay from prolonged moisture accumulation, likely from failed flashing or inadequate drainage at deck corner.
Immediate Action: Barricade the affected deck area immediately. Inspect subframe (joists, ledger board) for decay — probe with screwdriver. If subframe is compromised, full deck replacement required.
Recommended Timeline: Barricade immediately; structural assessment within 48 hours; deck repair or replacement within 4 weeks.
Estimated Cost Impact: $3,000–$15,000 depending on whether boards only or full deck structure requires replacement."""
    },
    {
        "instruction": "Water infiltration through the curtain wall glazing on the high-rise tower — happening on multiple floors during storm.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the leakage at the gasket/seal between glass and frame, at the corner joints, or at the interface between the curtain wall and the concrete slab edge?
2. Is the infiltration on one specific elevation (windward face) or multiple elevations?
3. How old is the building and has the curtain wall silicone ever been replaced or resealed?

DIAGNOSIS:
Root cause: Failed silicone sealant joints or gaskets in the curtain wall system allowing wind-driven rain to penetrate under pressure. High-rise curtain walls rely on pressure-equalized systems — once the outer seal fails, water can bypass the inner drainage cavity. Corner joints and slab edge interfaces are highest failure risk points.
Contributing factors: UV-degraded silicone (15–20 year life), thermal movement causing silicone fatigue, or original installation defects.

MANAGER SUMMARY
Severity: HIGH
Issue: Multi-floor curtain wall water infiltration during storms — systemic sealant failure in high-rise glazing system.
Root Cause: Aged or failed silicone sealant joints in curtain wall allowing wind-driven rain infiltration across multiple floors.
Immediate Action: Deploy internal drip protection on affected floors; engage curtain wall specialist (facade engineer) for rope-access inspection and test reports within 2 weeks.
Recommended Timeline: Facade inspection within 2 weeks; phased silicone replacement starting with highest-leak areas within 3–6 months.
Estimated Cost Impact: $50,000–$500,000+ for full curtain wall reseal depending on building height and area."""
    },
]

# ---------------------------------------------------------------------------
# MINING DEFECTS — 30+ samples
# ---------------------------------------------------------------------------
MINING_SAMPLES = [

    # ── Conveyor Belt Tear ─────────────────────────────────────────────────
    {
        "instruction": "There's a longitudinal tear running along the centre of the conveyor belt, about 2 metres long. Belt is still running.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the tear on the carrying (top) surface, the pulley (bottom) surface, or has it penetrated through the full belt thickness?
2. What material is being conveyed, and at what speed and belt tension is the system operating?
3. Is there any known sharp object, tramp metal, or structural obstruction in the load zone that could have caused the tear?

DIAGNOSIS:
Root cause: Longitudinal belt tear is most commonly caused by tramp metal or sharp foreign objects (bolts, rock shards) jamming in a belt cleaner, chute, or structural frame and slicing the belt as it passes. A 2-metre longitudinal tear that runs along the centre line indicates the obstruction was at a point that the belt centre passes over — typically a belt cleaner holder or chute lip.
Risk: Continued operation with a longitudinal tear will propagate the tear rapidly, potentially leading to belt splitting into two sections — catastrophic loss of the belt and production stoppage. The tear can also catch on equipment and cause injury.

MANAGER SUMMARY
Severity: CRITICAL
Issue: 2-metre longitudinal conveyor belt tear — critical risk of full belt failure and equipment damage if operation continues.
Root Cause: Tramp metal or structural obstruction in load zone or at belt cleaner causing the belt to be sliced longitudinally.
Immediate Action: STOP the conveyor immediately. Lock out/tag out (LOTO). Inspect belt cleaner, chutes, and load zone for tramp metal or obstruction. Assess belt for full-thickness penetration. Do not restart until cause is identified and removed.
Recommended Timeline: LOTO and inspection now; belt vulcanisation repair or emergency replacement within 24–48 hours; tramp metal detector installation recommended.
Estimated Cost Impact: $3,000–$15,000 for belt vulcanisation; $20,000–$100,000+ for belt replacement depending on belt width and length."""
    },
    {
        "instruction": "The conveyor belt has a transverse cut across its width, about 30cm long, looks like something hit it.",
        "response": """FOLLOW-UP QUESTIONS:
1. Where on the belt is the transverse cut relative to the system — in the load zone, under a transfer chute, or in the return section?
2. Has there been any recent maintenance where tools, bolts, or components could have dropped onto the belt?
3. Is the cut through the full belt thickness or just the top rubber cover layer?

DIAGNOSIS:
Root cause: Impact damage from a dropped object (maintenance tool, rock, structural component) causing a transverse impact cut. Transverse cuts are less dangerous than longitudinal tears as they do not propagate along the belt direction under tension. However, a full-thickness cut is a potential belt failure point at the weakest cross-section.
Risk assessment: If the cut is cover-only, the belt carcass (fabric/steel cord) is intact and the risk is lower — monitor and plan scheduled repair. If the cut penetrates the carcass, belt tensile strength is compromised at that point.

MANAGER SUMMARY
Severity: HIGH
Issue: Transverse belt cut — potential belt failure if carcass penetrated; requires assessment and repair.
Root Cause: Impact damage from dropped object onto belt surface.
Immediate Action: LOTO and inspect cut depth — probe with knife to check carcass penetration. If carcass is cut, plan emergency repair. Mark the damaged section with paint or tape for tracking.
Recommended Timeline: Immediate LOTO inspection; vulcanisation repair within 24 hours if carcass damaged; cover repair within scheduled maintenance window if cover-only.
Estimated Cost Impact: $500–$3,000 for cover repair; $5,000–$20,000 for full carcass repair or splice."""
    },

    # ── Belt Wear ──────────────────────────────────────────────────────────
    {
        "instruction": "The belt rubber cover is worn through in the centre, I can see the fabric carcass exposed over about a 5-metre section.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the wear in the centre of the belt (indicating abrasive load impact from the chute centre) or at the edge (indicating misalignment or edge roller wear)?
2. What material is being conveyed — coarse rock, fine ore, coal, or aggregate?
3. When was the belt last replaced or turned (flipped), and what is the belt's total operating hours?

DIAGNOSIS:
Root cause: Abrasive wear of the carrying cover from high-impact, abrasive material loading. The exposed fabric carcass over a 5-metre section indicates the cover has been worn to zero thickness. This typically happens in the impact zone under the load chute where large lumps fall and abrade the belt with each cycle. The centre-line concentration suggests the chute stream is centred but impacting with insufficient chute sealing.
Risk: Exposed fabric carcass is vulnerable to moisture absorption (causing delamination), direct abrasion (accelerating further), and will fail rapidly without intervention.

MANAGER SUMMARY
Severity: HIGH
Issue: Exposed fabric carcass over 5-metre section of conveyor belt — accelerating deterioration and imminent belt failure risk.
Root Cause: Abrasive wear from high-impact material loading at chute point wearing cover beyond minimum thickness.
Immediate Action: Reduce load rate or material lump size if possible. Apply emergency belt cover repair compound or patch to protect carcass. Plan belt replacement.
Recommended Timeline: Emergency cover patch within 24 hours; schedule belt replacement within 2–4 weeks during planned shutdown; install chute impact cradles or deceleration zone to reduce future wear.
Estimated Cost Impact: $500–$1,500 emergency patch; $10,000–$80,000 for belt replacement; $5,000–$15,000 for chute improvement."""
    },
    {
        "instruction": "Belt edges are fraying and the rubber edge strip is peeling off on one side for about 10 metres.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the fraying occurring on both edges or only one edge, and does the affected edge correspond to the same side along the full belt run?
2. Are there any idler rollers on the affected edge that appear to be out of alignment or running hot?
3. Has the belt recently been tracking off-centre, or are there signs of the belt running against the structure?

DIAGNOSIS:
Root cause: Belt mistracking causing the belt edge to run against a structural frame, chute side, or skirtboard, abrading and peeling the edge rubber. Single-side fraying concentrated on one side indicates the belt has been running offset to that side. This is caused by idler misalignment, uneven loading (off-centre chute stream), or a bent idler frame forcing the belt to one side.
Risk: Progressive edge fraying weakens the belt, can cause belt edge to catch on structure (leading to belt tear), and material spillage from reduced belt width.

MANAGER SUMMARY
Severity: MEDIUM
Issue: Conveyor belt edge fraying and rubber strip separation on one side over 10 metres — indicates mistracking requiring correction.
Root Cause: Belt mistracking causing edge contact with structure or skirtboard; likely caused by idler misalignment or off-centre loading.
Immediate Action: Walk the full conveyor run and identify any idlers that are visibly out of alignment or canted; check load chute centreline position; adjust any misaligned idlers.
Recommended Timeline: Idler alignment correction within 48 hours; edge damage repair within 1 week; install training idlers if mistracking persists.
Estimated Cost Impact: $200–$800 idler adjustment; $2,000–$6,000 edge repair; $3,000–$8,000 for training idler installation."""
    },

    # ── Belt Deviation ─────────────────────────────────────────────────────
    {
        "instruction": "The conveyor belt keeps drifting to the left side when loaded and coming back to centre when empty.",
        "response": """FOLLOW-UP QUESTIONS:
1. At what point along the conveyor does the belt first start drifting left — near the tail, in the middle, or near the head pulley?
2. Is the load distribution on the belt centred, or does material appear to be loading off to one side of the chute?
3. Are the return strand idlers in the drift section horizontal (flat) or angled (V-return)?

DIAGNOSIS:
Root cause: The belt drifts under load but centres when empty — this is the classic signature of off-centre loading. When material loads off-centre (to the right in this case), the weight distribution shifts the belt to the left as the unloaded right side has less down-force on the idlers. The belt self-corrects when empty because the load asymmetry is removed.
Secondary check: If loading is centred, the drift may be caused by a canted idler that only produces lateral force under the increased friction of a loaded belt.

MANAGER SUMMARY
Severity: MEDIUM
Issue: Belt drifting to left under load — off-centre material loading or canted idler causing belt mistracking risk.
Root Cause: Off-centre material loading from chute misalignment or canted idler frame producing lateral belt force under load.
Immediate Action: Observe chute loading point — check if material stream is centred on the belt. If off-centre, adjust chute discharge direction or install loading guides. Check idler frames for canting in the drift zone.
Recommended Timeline: Loading adjustment within 24 hours; if idler alignment required, complete within 1 week; install load-sensing training idlers if problem persists.
Estimated Cost Impact: $500–$2,000 chute or idler adjustment; $3,000–$8,000 if structural chute modification required."""
    },
    {
        "instruction": "Belt has been running off to the right at the tail end and wearing against the chute structure.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the tail pulley crowned (convex profile) or flat, and when was it last inspected?
2. Are the tail idlers in alignment and parallel to the tail pulley?
3. Has the belt recently been spliced or has a new belt section been installed?

DIAGNOSIS:
Root cause: Belt deviation at the tail end is often caused by a tail pulley that is not square (perpendicular) to the belt centreline, or tail idlers that are misaligned. If the tail pulley is canted or the belt approaches the tail off-angle (due to a canted transition idler), the belt will consistently track to one side at the tail. A new belt splice can also cause deviation if the splice is not square.

MANAGER SUMMARY
Severity: MEDIUM
Issue: Belt deviation against chute structure at tail end — causing belt edge damage and potential structural contact wear.
Root Cause: Tail pulley misalignment or tail idler canting causing belt to track consistently right.
Immediate Action: LOTO; check tail pulley alignment with a straight edge and level; check tail idler frame parallelism; adjust as required.
Recommended Timeline: Alignment check within 24 hours; adjustment within 48 hours; monitor for 1 week post-adjustment.
Estimated Cost Impact: $200–$1,000 for alignment adjustment; $2,000–$5,000 if pulley bearing housing requires repositioning."""
    },

    # ── Idler Roller Faults ───────────────────────────────────────────────
    {
        "instruction": "One of the idler rollers on the return strand is seized and not rotating. There's a flat spot on it now.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the seized roller in the carrying strand (top) or return strand (bottom), and how long has it been seized?
2. Is there visible heat discolouration (blue/brown) on the roller, or any smoke/burning smell?
3. Is the roller one of a set of three in a troughing frame or a single flat roller?

DIAGNOSIS:
Root cause: Idler roller bearing failure causing seizure. Return strand rollers are exposed to spillage material accumulating on the roller and in the bearing seals, accelerating bearing wear. A seized roller is not merely an efficiency issue — it creates a flat spot that hammers the belt with each revolution, and a seized roller can generate sufficient heat through friction to ignite conveyor belt material (fire risk in coal or other combustible mines).
Risk: Fire hazard from friction heat if the belt is running over the seized flat-spot roller.

MANAGER SUMMARY
Severity: HIGH
Issue: Seized return idler roller with flat spot — fire risk from friction heat and progressive belt damage.
Root Cause: Bearing failure in return strand idler allowing roller seizure; accelerated by spillage material contaminating bearing seals.
Immediate Action: LOTO; replace seized roller immediately; inspect belt underside for heat damage or wear marks; check for smoke or heat from the roller position.
Recommended Timeline: Roller replacement within 4 hours; inspect all return rollers in the same section within 24 hours.
Estimated Cost Impact: $50–$300 per roller; $2,000–$5,000 if belt underside damage requires repair."""
    },
    {
        "instruction": "Can hear a loud grinding/squealing noise from the idler section under the load zone. Can't see which roller it is.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the noise constant or does it pulse rhythmically (once per roller revolution)?
2. Does the noise change in pitch or intensity when the belt speed is changed?
3. Is the noise coming from the top carrying strand or the return strand under the belt?

DIAGNOSIS:
Root cause: Bearing failure in one or more idler rollers causing metal-on-metal grinding as the bearing race breaks up. The grinding/squealing is the characteristic sound of a bearing running without lubrication or with shattered race elements. Under the load zone, the high downforce accelerates failure once a bearing begins to wear.
Identification method: Run a contact thermometer or thermal camera along the idlers — the failed bearing will run significantly hotter (50°C+ above ambient). A rubber mallet tap on each roller while running can also help localise the seized one by sound change.

MANAGER SUMMARY
Severity: HIGH
Issue: Grinding/squealing from load zone idlers — bearing failure in progress, risk of roller seizure and belt damage.
Root Cause: Advanced bearing failure in one or more load zone idlers — metal-on-metal contact from depleted lubrication or race failure.
Immediate Action: Use thermal camera or contact thermometer to identify the hot roller; replace immediately under LOTO.
Recommended Timeline: Identification and replacement within 4 hours; inspect all load zone rollers thermally within 24 hours.
Estimated Cost Impact: $50–$300 per roller replacement; $500–$1,000 for thermal inspection campaign."""
    },

    # ── Tunnel Lining Cracks ───────────────────────────────────────────────
    {
        "instruction": "Crack in the concrete tunnel lining in the main haulage tunnel, about 1.5 metres long, running diagonally.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the crack wet (water seeping through it) or dry?
2. What is the approximate crack width — hairline (<0.2mm), visible gap (0.2–1mm), or finger-width (>1mm)?
3. Is the crack in the crown (top), shoulder, or sidewall of the tunnel, and is there any visible displacement (one side higher than the other)?

DIAGNOSIS:
Root cause: Diagonal crack in tunnel lining indicates shear stress from rock mass pressure distribution. In mining tunnels, diagonal cracks commonly result from: asymmetric rock load (higher pressure from one side), blast-induced micro-fractures in the surrounding rock causing load redistribution, or tunnel geometry that creates stress concentration at corners.
Critical assessment needed: A crack wider than 0.5mm with water seepage and displacement requires immediate structural assessment — it may indicate active rock mass movement.

MANAGER SUMMARY
Severity: HIGH
Issue: 1.5-metre diagonal crack in main haulage tunnel lining — potential indicator of rock mass stress redistribution requiring structural assessment.
Root Cause: Shear stress from asymmetric rock mass pressure or blast-induced disturbance causing diagonal tension-shear cracking in tunnel lining.
Immediate Action: Install crack monitoring tell-tales (glass tells or displacement pins) immediately; measure and record crack width; report to mine geotechnical engineer for assessment. If crack is >1mm wide or displacing, consider temporary restriction of tunnel access.
Recommended Timeline: Geotechnical engineer inspection within 24 hours; crack monitoring readings at 6-hour intervals for 48 hours; remediation (shotcrete, rock bolts, or gunite) per engineer's recommendation.
Estimated Cost Impact: $2,000–$5,000 monitoring and assessment; $10,000–$80,000+ for structural remediation depending on extent."""
    },
    {
        "instruction": "Multiple cracks at the crown of the tunnel, some with white mineral deposits on them. Tunnel is about 15 years old.",
        "response": """FOLLOW-UP QUESTIONS:
1. Are the cracks at the crown running longitudinally (along the tunnel axis) or transversely (across the tunnel)?
2. How widespread is the cracking — a short section of 5–10 metres, or does it extend for a long stretch?
3. Have there been any recent blasting operations in nearby headings or development faces?

DIAGNOSIS:
Root cause: Crown cracking with white mineral deposits (calcite or lime leaching) indicates long-term water seepage through the lining, dissolving calcium carbonate from the concrete and redepositing it as white stalactites or deposits on the crack faces. After 15 years, this is a combination of: initial concrete shrinkage cracking, cyclic loading from blasting vibration, and water chemistry deterioration of the concrete.
The mineral deposits confirm these cracks have been actively seeping for an extended period, potentially causing rebar corrosion within the lining.

MANAGER SUMMARY
Severity: HIGH
Issue: Multiple mineral-stained crown cracks in 15-year-old tunnel — long-term water seepage indicates degraded lining integrity and potential rebar corrosion.
Root Cause: Long-term water infiltration through aging tunnel lining cracks dissolving concrete minerals; rebar corrosion risk within the lining.
Immediate Action: Install monitoring points; engage geotechnical/structural engineer for comprehensive tunnel lining inspection; conduct rebar cover depth and corrosion potential measurements.
Recommended Timeline: Engineering inspection within 1 week; full condition survey within 1 month; rehabilitation planning within 3 months.
Estimated Cost Impact: $10,000–$50,000 condition survey; $50,000–$500,000+ for tunnel lining rehabilitation depending on extent."""
    },

    # ── Tunnel Seepage ────────────────────────────────────────────────────
    {
        "instruction": "Water dripping from the tunnel wall at a joint between two lining segments. Flow rate is about a steady drip.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the water clear, turbid (cloudy/silty), or coloured (red/brown from iron, white from lime)?
2. Does the flow rate increase after rain events or is it constant regardless of surface weather?
3. Is there any sign of ground movement — displaced joint faces, debris at the joint, or nearby cracking?

DIAGNOSIS:
Root cause: Failed joint sealant or gasket between tunnel lining segments allowing groundwater to seep through. Segmental tunnel linings rely on gasket seals (EPDM or neoprene) at every joint — these seals can fail due to age, chemical degradation, or differential segment movement.
Water chemistry clues: Clear water = groundwater seeping through failed gasket; turbid water = soil fines being washed into tunnel (piping risk — critical); brown/red = iron-rich groundwater; white deposits = calcium carbonate dissolution.

MANAGER SUMMARY
Severity: MEDIUM
Issue: Active water seepage at tunnel lining segment joint — joint gasket failure allowing groundwater ingress.
Root Cause: Failed joint gasket or sealant at segmental lining joint.
Immediate Action: Measure and record flow rate and water turbidity; if turbid, escalate to CRITICAL — turbid seepage indicates ground fines migration (piping). For clear water, grout injection repair can be planned.
Recommended Timeline: Assessment within 24 hours; polyurethane grout injection repair within 1–2 weeks for clear seepage; immediate CRITICAL response for turbid seepage.
Estimated Cost Impact: $500–$3,000 per injection point for grout sealing; $5,000–$20,000 for comprehensive joint sealing campaign."""
    },
    {
        "instruction": "Large volume of water coming through the floor at the base of the tunnel after overnight rain. Never happened before.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the water coming through the floor slab as seepage, or is it coming from a specific crack or joint?
2. What is the approximate depth of this tunnel below the surface, and is there any surface drainage (creek, drain) nearby?
3. Has any new excavation, drilling, or blasting been done in the area in the past week?

DIAGNOSIS:
Root cause: Sudden high-volume water inrush after rain is a serious event — likely caused by a failure of the groundwater barrier, either through the invert (floor), at a joint, or through a pre-existing fracture network that has connected to a surface water source. The fact it has never happened before and correlates with rainfall suggests a new flow path has been opened — potentially by subsidence, blasting fractures connecting to surface, or an overloaded drainage system backing up.
Emergency risk: High-volume water inrush can rapidly flood a tunnel, creating a life-safety emergency.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Sudden high-volume water inrush through tunnel floor after rain — potential tunnel flooding risk and life-safety emergency.
Root Cause: New groundwater flow path opening to surface water (possibly blasting-induced fracture or subsidence crack connecting to surface drainage).
Immediate Action: EVACUATE tunnel personnel immediately. Activate mine emergency response plan. Engage mine water management and geotechnical team. Deploy pumps if safe to do so from outside the tunnel.
Recommended Timeline: Evacuation now; emergency pumping within 1 hour; hydrogeological investigation within 24 hours; no re-entry until source identified and controlled.
Estimated Cost Impact: $10,000–$100,000+ emergency response; production stoppage costs; $20,000–$200,000 for investigation and sealing."""
    },

    # ── Hydraulic Oil Leaks ───────────────────────────────────────────────
    {
        "instruction": "Hydraulic hose on the excavator boom cylinder is leaking oil. Oil is spraying in a fine mist when the cylinder extends.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the spray coming from the hose fitting/crimp end, or from a hole/abrasion in the middle of the hose?
2. What is the system operating pressure — is this a high-pressure main line (>200 bar) or a return/pilot line?
3. Is the spray directed toward any hot surfaces (exhaust, turbo) or electrical connections?

DIAGNOSIS:
Root cause: Hydraulic hose failure — either a crimp fitting separation (fitting has pulled out of the hose end) or abrasion hole through the hose wall from rubbing against the boom structure. A fine mist spray under pressure is a critical hazard: high-pressure hydraulic oil misting can spontaneously ignite if it contacts a hot surface (exhaust, >200°C), causing hydraulic fire. Additionally, high-pressure oil injection into skin from a pin-hole leak can cause serious injection injuries.
Risk: Fire and injection injury — do not touch the mist with bare hands to locate the hole.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Hydraulic hose spray misting on excavator boom — fire and injection injury risk requiring immediate shutdown.
Root Cause: Failed hydraulic hose (crimp or abrasion failure) causing high-pressure oil mist spray near hot surfaces.
Immediate Action: SHUT DOWN excavator immediately. Move machine away from any ignition sources. LOTO. Do not attempt to touch or locate leak with hands (injection injury risk). Replace hose under LOTO before returning to service.
Recommended Timeline: Machine shutdown now; hose replacement within 4 hours; inspect all boom hoses for abrasion damage before restart.
Estimated Cost Impact: $200–$800 hose replacement; $50,000–$500,000+ potential cost if fire occurs."""
    },
    {
        "instruction": "Hydraulic pump on the longwall shearer is leaking from the shaft seal. Oil is coating the side of the pump housing.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the leak a slow seep (film of oil) or an active drip, and has the oil level in the hydraulic tank dropped noticeably?
2. When was the shaft seal last replaced, and what is the operating hours on the pump?
3. Is there any vibration or noise change from the pump that might indicate bearing wear causing shaft misalignment?

DIAGNOSIS:
Root cause: Hydraulic pump shaft seal failure — the dynamic lip seal around the rotating pump shaft has worn past its service life or has been damaged by shaft vibration (which causes eccentric shaft movement, destroying the seal lip). Shaft seal leaks in longwall equipment will progressively worsen and lead to contamination of the surrounding area.
In an underground coal environment, hydraulic oil leaks are a serious fire risk as oil can contact electricals or heat sources, and fire in a longwall panel is a major emergency.

MANAGER SUMMARY
Severity: HIGH
Issue: Hydraulic pump shaft seal leaking on longwall shearer — fire and equipment damage risk in underground coal environment.
Root Cause: Worn or damaged pump shaft seal — possibly accelerated by shaft vibration from bearing wear.
Immediate Action: Reduce pump load if possible; collect oil drip to prevent floor accumulation; schedule replacement at next planned maintenance window. Check bearings for excessive vibration/play.
Recommended Timeline: Shaft seal replacement within 24–48 hours during next maintenance window; bearing inspection concurrently.
Estimated Cost Impact: $500–$2,000 seal replacement; $5,000–$20,000 if pump rebuild required."""
    },

    # ── Equipment Oil Puddles ──────────────────────────────────────────────
    {
        "instruction": "Large oil puddle under the jaw crusher, roughly 20 litres based on puddle size. Equipment was idle overnight.",
        "response": """FOLLOW-UP QUESTIONS:
1. What type of oil is it — hydraulic (light golden/amber), gear oil (dark brown, thick), or engine oil (black/dark brown)?
2. Is the puddle directly under the flywheel bearing, under the toggle plate area, or under the crusher's lube oil tank?
3. Has the crusher shown any unusual temperatures, noise, or vibration in recent operating shifts?

DIAGNOSIS:
Root cause: A 20-litre overnight loss is a significant leak — not a minor seep. Jaw crusher oil puddles typically originate from: (a) flywheel bearing housing seal failure (gear/lubricant oil), (b) lube oil tank drain valve left open or fitting failure, or (c) toggle plate lubrication system failure. The location of the puddle under a specific component is the key diagnostic.
Risk: 20 litres overnight indicates the leak rate will cause rapid lubrication loss leading to bearing failure if not repaired.

MANAGER SUMMARY
Severity: HIGH
Issue: 20-litre overnight oil puddle under jaw crusher — significant active leak risk causing lubrication loss and bearing damage.
Root Cause: Major oil seal or fitting failure in crusher lubrication system — exact source requires inspection under LOTO.
Immediate Action: LOTO crusher; check lube oil levels before any attempt to restart; locate leak source; check bearings for temperature and noise on restart if levels are adequate.
Recommended Timeline: Inspection within 4 hours; repair within 24 hours; do not operate crusher until leak source identified and oil levels confirmed.
Estimated Cost Impact: $500–$3,000 seal replacement; $20,000–$100,000 if bearing damage from oil starvation has occurred."""
    },
    {
        "instruction": "Oil dripping from the gearbox on the secondary conveyor drive. The gearbox oil sight glass is reading low.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the leak at the output shaft seal, the input shaft seal, or from the gearbox body (possibly a crack or failed drain plug)?
2. What is the gearbox oil type and when was it last changed?
3. Has the gearbox been running at elevated temperature — is the oil level drop consistent with expected top-up intervals or is this sudden?

DIAGNOSIS:
Root cause: Gearbox shaft seal failure — the output or input shaft lip seal has worn through its service life, allowing gear oil to escape along the shaft. Gear oil on the output shaft side is common due to the weight of the shaft pulling down on the bottom lip seal. Low oil level with visible leak confirms active seal failure.
Risk: Continued operation with low oil level will cause gearbox gear and bearing damage from oil starvation — gearbox seizure is the failure mode, causing conveyor stoppage and expensive replacement.

MANAGER SUMMARY
Severity: HIGH
Issue: Gearbox oil leak with low oil level on secondary conveyor drive — risk of gearbox seizure from oil starvation.
Root Cause: Gearbox shaft seal failure allowing gear oil to escape; oil level now below minimum.
Immediate Action: LOTO; top up gearbox oil to correct level (with correct specification oil); identify and repair leak before restarting. Do not run the gearbox with low oil.
Recommended Timeline: Oil top-up and leak identification within 4 hours; shaft seal replacement within 24 hours at scheduled maintenance window.
Estimated Cost Impact: $300–$1,000 seal replacement; $5,000–$30,000 if gearbox requires rebuild due to gear/bearing damage."""
    },

    # ── Additional Mining Samples ─────────────────────────────────────────
    {
        "instruction": "The head pulley lagging on the main trunk conveyor is partially delaminated — can see the lagging peeling off in chunks.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the lagging ceramic tile type, diamond rubber, or plain rubber, and how old is it?
2. Is the peeling concentrated on one circumferential area of the pulley (indicating impact damage) or distributed around the full pulley?
3. Has there been increased belt slip on the drive recently, or belt tension warnings from the take-up?

DIAGNOSIS:
Root cause: Pulley lagging delamination from bond failure between the lagging and the pulley shell. Common causes: moisture ingress under the lagging (if operating in wet conditions), bond adhesive failure from age/chemical attack, or impact damage from belt tracking events. Delaminating lagging creates an uneven pulley surface — this causes belt mistracking, vibration, and belt damage on each revolution.
Risk: Belt slip and potential belt damage from lagging chunks catching on the belt or belt cleaners.

MANAGER SUMMARY
Severity: HIGH
Issue: Head pulley lagging delamination — risk of belt slip, mistracking, and belt damage from loose lagging chunks.
Root Cause: Bond failure of pulley lagging from moisture ingress, adhesive aging, or impact damage.
Immediate Action: Remove any fully delaminated chunks immediately (by hand with LOTO); monitor belt slip and tension; plan lagging replacement at next scheduled shutdown.
Recommended Timeline: Loose chunk removal under LOTO within 4 hours; full pulley re-lagging within 2–4 weeks at planned shutdown.
Estimated Cost Impact: $2,000–$8,000 for pulley re-lagging; $5,000–$20,000 if belt damage has occurred."""
    },
    {
        "instruction": "Belt scale on the main trunk conveyor is reading inconsistently — tonnage readings jump wildly between shifts.",
        "response": """FOLLOW-UP QUESTIONS:
1. Has anything changed on the conveyor recently — new belt splice, idler replacement, or belt tension adjustment?
2. Are the weight readings consistently high, consistently low, or genuinely erratic?
3. Is there material buildup or debris on the belt scale weigh frame or load cells?

DIAGNOSIS:
Root cause: Belt scale inaccuracy from load cell contamination (material buildup on the weigh frame altering the tare), a damaged or loose load cell, or a belt splice crossing the weigh zone creating a repeatable weight anomaly. Erratic readings across shifts suggest vibration or a marginal electrical connection to the load cell or indicator.
Impact: Inaccurate belt scale readings affect production reconciliation, ore grade tracking, and throughput reporting — financial and compliance implications.

MANAGER SUMMARY
Severity: MEDIUM
Issue: Belt scale giving erratic tonnage readings — production measurement accuracy compromised.
Root Cause: Load cell contamination, splice interference, or electrical connection fault in belt scale system.
Immediate Action: Clean the weigh frame and load cells; perform a static zero calibration; check all cable connections to the indicator; run a simulated load test.
Recommended Timeline: Calibration check within 24 hours; load cell replacement if required within 1 week; schedule monthly zero and span checks.
Estimated Cost Impact: $500–$2,000 calibration and servicing; $2,000–$5,000 load cell replacement if required."""
    },
    {
        "instruction": "Flywheel bolts on the jaw crusher are loose — I can see movement when the crusher is running. Machine is still operating.",
        "response": """FOLLOW-UP QUESTIONS:
1. How many flywheel bolts appear loose — one or two, or multiple bolts across the flange?
2. Is the flywheel itself visibly eccentric (wobbling) or is the movement just at the bolt heads?
3. How long has the crusher been operating since the last major service?

DIAGNOSIS:
Root cause: Flywheel bolt loosening from vibration fretting — the cyclic impact loads from jaw crushing cause bolts to back off if not properly torqued or if locking devices (tab washers, nord-lock washers) have failed. A loose flywheel is an extremely dangerous condition — flywheels weigh several tonnes and can detach catastrophically.
Risk: Catastrophic — a detached flywheel at operating speed is a fatal projectile hazard. This is a red-zone stop-work situation.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Loose flywheel bolts on running jaw crusher — catastrophic flywheel detachment risk, immediate life safety emergency.
Root Cause: Vibration-induced bolt loosening with failed or absent bolt locking device on flywheel flange.
Immediate Action: STOP THE CRUSHER IMMEDIATELY. Clear the area — exclusion zone minimum 10 metres. LOTO. Engage mechanical engineer/OEM to inspect and re-torque all flywheel bolts with correct locking system before any restart.
Recommended Timeline: Stop now; mechanical inspection within 4 hours; re-torque and lock within 24 hours; implement 8-hour bolt check during running-in period post-repair.
Estimated Cost Impact: $500–$2,000 re-torque and inspection; potentially millions in liability if flywheel detaches causing injury."""
    },
    {
        "instruction": "Underground pump station sump is filling with water faster than the pump can handle. Water level rising.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is this a gradual increase in water inflow over days, or a sudden step-change in inflow rate?
2. Has there been recent blasting in adjacent development headings or in an area above the pump station?
3. Is the pump running continuously at full capacity, and when was it last serviced or the impeller inspected?

DIAGNOSIS:
Root cause: Increased water inflow exceeding pump capacity — caused by either: (a) new water pathway opened by blasting or geological change, (b) surface water ingress after heavy rain connecting to underground via fractures, or (c) pump performance degradation from impeller wear reducing actual flow rate. Rising water in an underground pump station is a flooding risk.
Emergency protocol: If the sump level cannot be controlled, evacuation of the lower mine levels may be required under mine safety emergency procedures.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Underground pump station sump flooding — water inflow exceeding pump capacity, risk of mine inundation.
Root Cause: Increased water inflow from new geological pathway or blast-induced fractures, or degraded pump performance — investigation required.
Immediate Action: Activate mine water management emergency response. Deploy additional submersible pumps immediately. Alert mine management and mine emergency response team. Monitor water level continuously — trigger evacuation if rising rate cannot be controlled.
Recommended Timeline: Additional pumps deployed within 2 hours; hydrogeological investigation within 24 hours; pump inspection and service within 24–48 hours.
Estimated Cost Impact: $10,000–$50,000 emergency pumping; $50,000–$500,000+ if mine flooding occurs."""
    },
    {
        "instruction": "Dust suppression nozzles on the crusher discharge are clogged, excessive dust in the crushing bay.",
        "response": """FOLLOW-UP QUESTIONS:
1. Are the nozzles clogged with mineral scale (white deposits), fines compaction, or do they appear physically damaged?
2. What is the water supply pressure to the dust suppression system — is the pressure adequate at the nozzle inlet?
3. Has the water supply source changed recently (different bore water or different treatment)?

DIAGNOSIS:
Root cause: Dust suppression nozzle clogging from scale buildup (calcium carbonate from hard bore water) or fine particle clogging in nozzle orifice. Clogged nozzles in the crusher discharge zone create an immediate occupational health risk (respirable crystalline silica — RCS — if material contains silica) and potential explosion risk if material is combustible coal.
Health risk: Silica dust exposure causes silicosis — an irreversible and fatal lung disease. This is a mandatory reportable issue in most jurisdictions.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Dust suppression system failure in crushing bay — mandatory exposure controls not functioning; immediate silica/coal dust health risk.
Root Cause: Nozzle clogging from scale or fines blocking dust suppression water spray.
Immediate Action: Require all personnel in the crushing bay to wear appropriate RPE (P2/P3 respirator); reduce crusher throughput or stop if dust levels are excessive; clear nozzle blockages immediately (flush or replace nozzles).
Recommended Timeline: Nozzle repair within 4 hours; full system flush and nozzle inspection within 24 hours; install filtration/softener if scale is the cause.
Estimated Cost Impact: $200–$1,000 nozzle replacement; $2,000–$8,000 water treatment upgrade; potential regulatory fine if exposure records are not maintained."""
    },
    {
        "instruction": "Crack detected in the main frame of the underground bogger (LHD), running from the bucket pivot area towards the cab.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the crack a surface crack (weld bead cracking) or does it appear to penetrate through the full frame thickness?
2. Is the crack growing — have you marked the crack tips and observed growth over the last shift?
3. Has the machine recently experienced an unusual impact event — driving into a wall, dropped load, or hard stop?

DIAGNOSIS:
Root cause: Fatigue crack initiation in the LHD main frame at the high-stress concentration point near the bucket pivot. LHD frames are subject to cyclic bending and torsional loads that, over thousands of cycles, initiate fatigue cracks at stress risers (weld toes, geometry changes). A crack growing from the bucket pivot toward the cab is propagating through the highest-loaded structural section of the machine.
Safety risk: Main frame failure underground is a catastrophic event — the machine can collapse, trapping the operator and blocking the tunnel.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Main frame crack on underground LHD — catastrophic machine collapse risk with operator safety implications.
Root Cause: Metal fatigue crack initiated at bucket pivot stress concentration point under cyclic loading.
Immediate Action: REMOVE MACHINE FROM SERVICE IMMEDIATELY. Do not operate until frame has been assessed by OEM structural engineer. Weld repair by certified mine welding contractor required.
Recommended Timeline: Machine out of service now; OEM structural engineer assessment within 24 hours; certified weld repair within 48–72 hours; proof load test before return to service.
Estimated Cost Impact: $5,000–$20,000 for frame weld repair; production impact from machine downtime."""
    },
    {
        "instruction": "Smoke coming from the drive motor on the conveyor belt system. Motor is still running but smoking.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the smoke coming from the motor body (winding insulation burning) or from the shaft/coupling end (mechanical friction)?
2. Is the motor drive a direct-on-line start, soft starter, or VFD controlled, and has the thermal overload tripped recently?
3. Has the conveyor been running at full load, or is this during a startup with a heavily loaded belt?

DIAGNOSIS:
Root cause: Motor winding insulation failure (burning smell, acrid smoke from body) or mechanical friction (coupling, bearing — typically oily smoke from shaft end). A motor running while smoking indicates the thermal protection has failed to trip, or the thermal overload is set incorrectly. Continued operation will cause complete motor burnout and potential fire.
Emergency: A motor on fire underground or in a mining environment is a major emergency — fire suppression must be immediately available.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Smoking conveyor drive motor — active motor burnout in progress; fire risk.
Root Cause: Motor winding failure or bearing/coupling friction overheating — continuing to run risks motor fire.
Immediate Action: STOP THE CONVEYOR IMMEDIATELY via the emergency stop. LOTO. Confirm motor temperature with thermal camera if safe to approach. Have fire extinguisher ready. Do not restart until motor has been replaced and cause of failure investigated.
Recommended Timeline: Conveyor stop now; motor replacement within 24–48 hours; investigate root cause (overloading, VFD fault, bearing failure) before restart.
Estimated Cost Impact: $3,000–$15,000 motor replacement; $50,000–$500,000 if fire occurs."""
    },
    {
        "instruction": "The tail drum of the conveyor is vibrating excessively and making a thumping noise at regular intervals.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the thumping occurring once per revolution of the drum (about 1–3 seconds per thump depending on drum size and belt speed)?
2. Is there visible material buildup on the tail drum surface — does the drum appear out-of-round?
3. Have you checked the tail drum bearing housings for play or movement?

DIAGNOSIS:
Root cause: Regular thumping at once-per-revolution cadence is caused by material buildup (carry-back material stuck to drum surface) creating an eccentric out-of-round condition. As the high-spot contacts the belt, it creates an impact that propagates through the structure. This is the most common cause of tail drum vibration.
Alternative cause: Bent or damaged drum shell, or failed bearing causing egg-shaped drum rotation.
Risk: Excessive vibration accelerates bearing failure, fatigues conveyor structure, and hammers the belt causing premature wear.

MANAGER SUMMARY
Severity: MEDIUM
Issue: Excessive tail drum vibration with regular thumping — eccentric material buildup or drum damage causing belt hammering and bearing stress.
Root Cause: Most likely: carry-back material buildup on drum creating eccentric rotation. Secondary possibility: bent drum or bearing failure.
Immediate Action: LOTO; clean drum surface of all buildup; inspect drum for out-of-round or dents; check bearing play.
Recommended Timeline: Clean and inspect within 4 hours; install or improve belt cleaner (secondary cleaner) to prevent carry-back recurrence; replace bearings if play is found.
Estimated Cost Impact: $200–$800 cleaning; $1,000–$3,000 bearing replacement; $5,000–$15,000 if drum replacement required."""
    },
]

# ---------------------------------------------------------------------------
# Additional mixed samples to reach 60+
# ---------------------------------------------------------------------------
EXTRA_SAMPLES = [
    {
        "instruction": "Sealant around all the window frames on the north facade is cracking and shrinking. I can see gaps between the frame and the wall.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the sealant cracking across its full width (cohesive failure) or pulling away from one side — frame or wall — only (adhesive failure)?
2. Is the sealant a single line (butt joint) or does it fill a deep gap (the gap depth should ideally be half the joint width)?
3. What is the approximate age of the sealant, and has it been previously replaced?

DIAGNOSIS:
Root cause: Silicone or polyurethane sealant end-of-service-life degradation. Facade sealants typically have a 15–20 year life — UV exposure, thermal cycling, and joint movement exceeding design range cause cohesive cracking or adhesive failure. Gaps between frame and wall create direct water ingress paths at the most vulnerable junction of the building envelope.
Risk: Water infiltration through window sealant joints is one of the most common causes of internal water damage in commercial buildings.

MANAGER SUMMARY
Severity: MEDIUM
Issue: Cracked and failed window sealant across north facade — water infiltration risk at all window perimeters.
Root Cause: Age-related sealant degradation — cohesive cracking or adhesive failure from thermal cycling and UV exposure beyond design life.
Immediate Action: Inspect for water staining inside each window reveal; prioritise the worst gaps for temporary weatherproofing tape.
Recommended Timeline: Full facade sealant replacement within 3 months (schedule during dry weather); use movement-rated silicone with correct joint width-to-depth ratio.
Estimated Cost Impact: $3,000–$15,000 per facade elevation depending on window count; $500–$2,000 emergency patching."""
    },
    {
        "instruction": "Water is pooling on the flat roof — I can see a pond about 30cm deep near one corner.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the roof drain or scupper visible in the ponding area — is it blocked or submerged?
2. How long has the ponding been visible, and has the water level risen, stayed the same, or receded?
3. What is the roof membrane type — single-ply (TPO/EPDM), bitumen-felt built-up, or liquid-applied?

DIAGNOSIS:
Root cause: Blocked or inadequate roof drainage causing water ponding. Flat roofs should have maximum 1:80 falls to drains. A 30cm deep pond indicates either: a blocked drain outlet, collapsed drain pipe, or insufficient drainage capacity during the storm event. Prolonged ponding (>48 hours) accelerates membrane degradation, and the hydrostatic weight loads the structure.
Structural weight: 30cm of water = 300kg/m² — this may exceed the roof design live load and requires structural check.

MANAGER SUMMARY
Severity: HIGH
Issue: 30cm deep water ponding on flat roof — structural overloading risk and accelerated membrane damage.
Root Cause: Blocked roof drain or inadequate drainage capacity causing ponding.
Immediate Action: Clear roof drain immediately — safety-harnessed worker to remove debris from drain; if water cannot drain, check overflow scupper function.
Recommended Timeline: Drain clearance within 4 hours; structural engineer to confirm roof load capacity if ponding persists >24 hours; full drainage system inspection within 1 week.
Estimated Cost Impact: $200–$500 drain clearance; $2,000–$8,000 structural assessment and drainage upgrade if required."""
    },
    {
        "instruction": "The reinforced concrete retaining wall has a visible lean towards the roadway. It's about 4 metres high.",
        "response": """FOLLOW-UP QUESTIONS:
1. Approximately how much lean is visible — a few centimetres at the top, or can you measure the offset at the top versus the base?
2. Are there any cracks visible in the wall face, particularly horizontal cracks at mid-height?
3. Has there been any recent rainfall, fill placement behind the wall, or changes in the drainage behind the wall?

DIAGNOSIS:
Root cause: Retaining wall rotation/overturning movement — the wall is rotating forward (toward the road) under lateral earth pressure from the retained soil. This indicates: inadequate toe resistance (foundation bearing failure), loss of passive earth pressure at the toe, drainage failure causing hydrostatic buildup behind the wall, or inadequate original design for current soil conditions.
A leaning 4-metre RC retaining wall is an imminent collapse risk for people and vehicles below.

MANAGER SUMMARY
Severity: CRITICAL
Issue: 4-metre RC retaining wall leaning toward roadway — imminent collapse risk to traffic and pedestrians.
Root Cause: Overturning movement from lateral earth pressure exceeding wall resistance — drainage failure or foundation failure probable causes.
Immediate Action: IMMEDIATELY close road/footpath below the wall — minimum 5-metre exclusion zone. Erect barriers and signage. Notify local authority and engage structural/geotechnical engineer for emergency assessment within 4 hours.
Recommended Timeline: Road closure now; emergency engineering assessment within 4 hours; temporary propping if engineer deems safe to approach within 24 hours; permanent remediation per engineer's design.
Estimated Cost Impact: $5,000–$10,000 emergency assessment and propping; $30,000–$200,000+ for wall reconstruction."""
    },
    {
        "instruction": "The emergency exit door on Level 2 is not closing fully — there's a 5mm gap at the bottom edge when shut.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the door a fire-rated door (required to be self-closing and latching), or a standard emergency exit door?
2. Has the door frame been checked for squareness — is the gap consistent at the bottom or wider on one side?
3. Does the door latch engage when pulled closed manually, or does the latch not reach the strike plate?

DIAGNOSIS:
Root cause: Emergency exit door gap failure from either: door frame distortion (building movement or settlement causing the frame to go out of square), door hinge failure allowing the door to sag, worn door closer mechanism losing tension, or floor warping pushing the door bottom outward. A 5mm gap at the bottom of a fire door compromises its fire resistance rating — smoke and hot gases can pass through.
Regulatory: Fire doors must be self-closing and form a complete seal (no visible gaps) to maintain the fire compartmentation they are designed to provide.

MANAGER SUMMARY
Severity: HIGH
Issue: Fire-rated emergency exit door not sealing — 5mm gap compromises fire compartmentation and smoke containment.
Root Cause: Door frame distortion, hinge failure, or door closer malfunction causing incomplete door closure.
Immediate Action: Inspect hinge condition and door closer tension; check frame squareness with a level; adjust closer and hinges if possible; engage fire door specialist if adjustment does not resolve gap.
Recommended Timeline: Adjustment attempt within 24 hours; specialist repair within 48 hours if adjustment fails; update fire door inspection register.
Estimated Cost Impact: $200–$600 adjustment; $800–$2,500 door or frame replacement if required."""
    },
    {
        "instruction": "Rock bolt inspection in the decline shows 3 bolts where the plate has popped off and the bolt is protruding with no restraint at the collar.",
        "response": """FOLLOW-UP QUESTIONS:
1. What type of rock bolts are these — resin-anchored rebar, mechanical shell bolts, or friction bolts (Split Set/Swellex)?
2. Is the surrounding rock at these bolt locations showing signs of rock mass movement — fresh fractures, spalling, or displacement of the shotcrete around the collar?
3. Were these bolts installed as part of the primary support pattern, or as secondary or rehabilitation bolts?

DIAGNOSIS:
Root cause: Rock bolt face plate failure — the plate has been ejected from the bolt due to excessive rock mass convergence (the rock has moved more than the bolt design tolerance), corrosion failure at the plate/washer, or incorrect installation (inadequate plate seating). In a decline, where vehicles and personnel travel, ejected bolt hardware and loose collars are a direct projectile and fall-of-ground hazard.
Geotechnical concern: Three bolts in the same area with plate ejection indicates localised rock mass instability — this zone may require rehabilitation.

MANAGER SUMMARY
Severity: CRITICAL
Issue: 3 rock bolts with ejected face plates in mine decline — fall-of-ground and projectile hazard in active traffic area.
Root Cause: Rock mass convergence exceeding bolt design tolerance causing face plate ejection, or corrosion/installation failure at the collar.
Immediate Action: CLOSE the decline to all personnel and vehicle traffic. Mark the affected zone. Engage mine geotechnical engineer for assessment within 4 hours. Do not re-enter until geotechnical clearance is given.
Recommended Timeline: Geotechnical assessment today; rehabilitation support design within 24 hours; additional rock support installation within 48 hours before re-opening.
Estimated Cost Impact: $5,000–$20,000 rehabilitation support; production impact from decline closure."""
    },
    {
        "instruction": "Waterproof membrane on the podium level has bubbles forming across multiple areas — at least 10 separate blisters visible.",
        "response": """FOLLOW-UP QUESTIONS:
1. What is the membrane type — torch-on bitumen, cold-applied liquid membrane, or sheet-applied HDPE/TPO?
2. Are the blisters in areas that get full sun exposure, or also in shaded areas?
3. How old is the membrane, and was it installed directly over the slab or over an insulation layer?

DIAGNOSIS:
Root cause: Multiple blistering indicates widespread entrapped moisture under the membrane — either moisture was present in the slab at time of membrane installation (concrete must be dry to <75% RH for most membranes), or water has infiltrated under the membrane edges and is trapped. Solar heating drives trapped moisture vapour pressure above the membrane bond strength, lifting the membrane from below.
Risk: Blistered membranes have zero waterproofing function in the blistered areas — water can pool and flow under the membrane to areas far from the blisters, causing internal water damage below the slab.

MANAGER SUMMARY
Severity: HIGH
Issue: Widespread podium membrane blistering across 10+ areas — systemic membrane failure with active water infiltration risk.
Root Cause: Entrapped moisture beneath membrane (wet substrate at time of installation, or edge infiltration) causing hydrostatic blistering under solar heating.
Immediate Action: Engage waterproofing specialist for condition assessment; do not attempt to puncture blisters without specialist guidance (can accelerate failure).
Recommended Timeline: Specialist assessment within 1 week; full membrane strip and replacement within 3–6 months (schedule for dry season); ensure slab moisture content is below specification before new membrane application.
Estimated Cost Impact: $2,000–$5,000 assessment; $30,000–$150,000+ full podium membrane replacement."""
    },
    {
        "instruction": "Pipe insulation on a chilled water pipe is wet and the pipe has condensation dripping off it in the plant room.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the insulation covering intact but wet on the outside, or is there a gap, tear, or missing section of insulation?
2. Is the dripping water clear (condensation) or has it a slight coloration or odour (pipe leak through insulation)?
3. What is the current plant room humidity level, and is the plant room air conditioning functioning?

DIAGNOSIS:
Root cause: Failed pipe insulation vapour barrier allowing warm humid plant room air to contact the cold chilled water pipe surface, condensing on the cold metal. Chilled water pipes operate below the dew point of the surrounding air — without an intact vapour barrier, condensation forms within and on the insulation, saturating it. Wet insulation loses its thermal value (wet insulation has ~1% of dry insulation thermal resistance) causing further temperature drop and more condensation — a self-reinforcing cycle.
Risk: Dripping water causes floor corrosion, slip hazard, and potential damage to electrical equipment below.

MANAGER SUMMARY
Severity: MEDIUM
Issue: Failed chilled water pipe insulation causing condensation dripping — thermal efficiency loss and slip/corrosion hazard.
Root Cause: Vapour barrier failure in pipe insulation allowing condensation within insulation material.
Immediate Action: Deploy drip trays under the dripping section; identify and isolate insulation damage; dry out before repair.
Recommended Timeline: Drip containment today; insulation replacement within 2 weeks; use closed-cell foam insulation with aluminium foil vapour barrier.
Estimated Cost Impact: $500–$3,000 per pipe section for insulation replacement; $200–$500 for drip management."""
    },
    {
        "instruction": "Scrubber water pH in the coal wash plant has been dropping — readings show pH 3.2 this morning, normal is pH 7–8.",
        "response": """FOLLOW-UP QUESTIONS:
1. Has there been any change in the coal feed — different seam, blending ratio, or higher sulphur content?
2. Is the pH dropping uniformly across all scrubber circuits or only on one specific circuit?
3. Has the lime dosing system been checked — is the lime slurry density and feed rate correct?

DIAGNOSIS:
Root cause: Acid mine drainage (AMD) or pyrite oxidation in the feed coal is generating sulphuric acid that is overwhelming the lime neutralisation capacity of the scrubber circuit. High sulphur coals contain pyrite (FeS₂) which oxidises during washing to produce sulphuric acid. pH 3.2 indicates the lime dosing system is no longer providing adequate neutralisation.
Environmental risk: Discharge of acidic process water at pH 3.2 is a serious environmental compliance breach and may require immediate cessation of discharge.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Scrubber water pH at 3.2 — severe acid condition indicating lime neutralisation failure; environmental discharge compliance breach.
Root Cause: Pyrite oxidation generating sulphuric acid exceeding current lime dosing capacity — possible high-sulphur feed coal.
Immediate Action: STOP discharge of process water immediately. Increase lime dosing rate. Collect water samples for analysis. Notify environmental officer and assess if regulatory notification is required.
Recommended Timeline: Dosing adjustment within 1 hour; laboratory analysis results within 4 hours; review coal feed quality and adjust lime dosing recipe within 24 hours.
Estimated Cost Impact: $5,000–$20,000 lime dosing adjustment; potential $50,000–$500,000 environmental fines if breach is prosecuted."""
    },
    {
        "instruction": "Vibration levels on the SAG mill pinion bearing have increased significantly over the past week. Readings are now at alarm threshold.",
        "response": """FOLLOW-UP QUESTIONS:
1. Which bearing position is showing the highest vibration — drive-side or non-drive-side pinion bearing?
2. Is the dominant frequency at 1× running speed (imbalance), gear mesh frequency (gear wear), or bearing defect frequencies?
3. Has there been any recent change in mill throughput, feed size, or ball charge?

DIAGNOSIS:
Root cause: Elevated bearing vibration at the alarm threshold indicates developing bearing damage — likely early-stage outer or inner race fatigue cracking, or gear mesh irregularity from gear wear/tooth damage. SAG mill pinion bearings operate under enormous loads, and vibration trending is the primary early warning system for impending failure.
Risk: If vibration continues to increase and the bearing fails catastrophically, the pinion can drop into the girth gear, causing catastrophic gear damage — a multi-week production stoppage.

MANAGER SUMMARY
Severity: HIGH
Issue: SAG mill pinion bearing vibration at alarm threshold — risk of catastrophic bearing and gear failure.
Root Cause: Developing bearing defect (race fatigue or cage damage) or gear mesh issue — frequency analysis required for definitive diagnosis.
Immediate Action: Increase vibration monitoring frequency to every 2 hours; schedule oil analysis; engage rotating equipment specialist for vibration signature analysis; plan bearing replacement at next available shutdown window.
Recommended Timeline: Vibration analysis within 24 hours; bearing replacement within 1–2 weeks if confirmed defect; do not delay beyond first available planned shutdown.
Estimated Cost Impact: $10,000–$30,000 planned bearing replacement; $200,000–$2,000,000 if gear and pinion failure occurs."""
    },
    {
        "instruction": "Concrete floor slab in the warehouse has a crack running 8 metres across the floor — it appeared overnight with no obvious cause.",
        "response": """FOLLOW-UP QUESTIONS:
1. Is the crack straight (following aggregate lines — shrinkage) or irregular/jagged (structural)?
2. Is there any vertical displacement — can you feel one side of the crack is higher than the other with your foot?
3. Has any new heavy plant, racking, or stored material been added to the warehouse in the past week?

DIAGNOSIS:
Root cause: Floor slab cracking can result from: (a) late-stage concrete shrinkage if the slab is relatively new (<2 years), (b) ground subsidence or sinkhole formation below the slab causing loss of support, or (c) overloading from new plant/racking exceeding the slab design capacity. An overnight appearance suggests either a sudden loading change or an acute ground movement event rather than gradual shrinkage.
Risk: If there is vertical displacement, this is a structural slab failure — the slab has lost support and the crack will propagate, potentially causing a slab collapse under loaded racking.

MANAGER SUMMARY
Severity: HIGH
Issue: 8-metre slab crack appeared overnight in warehouse — potential ground subsidence or overload causing structural slab failure.
Root Cause: Sudden ground movement, void formation below slab, or overloading — distinguished by displacement and loading history.
Immediate Action: Clear all racking and heavy loads from the affected bay; engage structural and geotechnical engineer for assessment within 48 hours; check if there are any services (water mains, drainage) below the slab that may have leaked and undermined the subgrade.
Recommended Timeline: Engineering assessment within 48 hours; CCTV drain inspection within 1 week; repair per engineer recommendation.
Estimated Cost Impact: $5,000–$15,000 investigation; $20,000–$100,000+ slab repair."""
    },
    {
        "instruction": "Scaffold on the façade of the building has one standard (vertical pole) that is visibly bent and leaning. Workers are currently using the scaffold.",
        "response": """FOLLOW-UP QUESTIONS:
1. How much is the standard leaning — a few degrees or visibly bowing more than 10mm out of vertical?
2. Is the bent standard a loaded standard (carrying ledger and board load) or a guard-rail-only standard?
3. When was the scaffold last inspected and signed off by a licensed scaffolding inspector?

DIAGNOSIS:
Root cause: Bent scaffold standard indicates overloading beyond the tube's compression capacity (buckling), a vehicle strike, material drop impact, or incorrect assembly using non-standard tube. A bent vertical standard has dramatically reduced load capacity — a buckled column can fail suddenly without further warning when loaded.
Regulatory: In most jurisdictions, scaffold with a visibly damaged member must be immediately taken out of service until the member is replaced and the scaffold is re-inspected.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Bent scaffold standard with workers on scaffold — immediate collapse risk and regulatory non-compliance.
Root Cause: Structural overload or impact damage causing scaffold standard to buckle, compromising structural integrity of the affected bay.
Immediate Action: EVACUATE scaffold immediately. Close access to the scaffold. Notify scaffold supervisor and engage licensed scaffolding contractor to replace the standard and re-inspect before any re-use.
Recommended Timeline: Evacuation now; licensed inspection and repair within 4 hours; full scaffold re-inspection and re-certification before return to use.
Estimated Cost Impact: $500–$1,500 repair; regulatory fine risk if workers are allowed to continue on unsafe scaffold."""
    },
    {
        "instruction": "Steel handrail on the mine access stairs is heavily corroded and has visible rust pitting. Some sections wobble when grabbed.",
        "response": """FOLLOW-UP QUESTIONS:
1. How many handrail sections are loose — individual posts wobbling, or the entire rail length moving?
2. Is the corrosion surface rust only (can be scraped to reveal sound metal), or is the steel section reduced in thickness (through-section pitting)?
3. Are the handrail posts welded to the stair structure or bolted with base plates?

DIAGNOSIS:
Root cause: Advanced galvanic or general corrosion of mild steel handrail without adequate protective coating in a high-humidity mining environment. Pitting corrosion reduces the cross-sectional area of the handrail posts, reducing their load-bearing capacity. Loose posts indicate either base weld failure from corrosion undercut or base plate bolt corrosion/failure.
Safety risk: A failed handrail during a fall is catastrophic — this is a critical safety item under mine safety regulations.

MANAGER SUMMARY
Severity: CRITICAL
Issue: Corroded and loose mine stairway handrail — regulatory non-compliance and fall hazard requiring immediate make-safe.
Root Cause: Advanced corrosion of mild steel handrail from exposure to mine humidity/chemicals — structural capacity of loose sections is unknown.
Immediate Action: Close affected stair section and post warning signs immediately; test all handrail sections for stability; replace all loose/heavily corroded sections before re-opening.
Recommended Timeline: Close and test within 4 hours; replacement of failed sections within 24–48 hours; full handrail replacement in hot-dip galvanized steel within 4 weeks.
Estimated Cost Impact: $2,000–$8,000 per stair flight for full replacement; $500–$2,000 for spot section replacement."""
    },
]

# ---------------------------------------------------------------------------
# Combine and write
# ---------------------------------------------------------------------------
ALL_SAMPLES = BUILDING_SAMPLES + MINING_SAMPLES + EXTRA_SAMPLES


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for sample in ALL_SAMPLES:
            line = json.dumps(sample, ensure_ascii=False)
            f.write(line + "\n")

    print(f"✅ Dataset written to: {OUTPUT_PATH}")
    print(f"   Total samples: {len(ALL_SAMPLES)}")
    print(f"   Building samples: {len(BUILDING_SAMPLES)}")
    print(f"   Mining samples:   {len(MINING_SAMPLES)}")
    print(f"   Extra samples:    {len(EXTRA_SAMPLES)}")


if __name__ == "__main__":
    main()
