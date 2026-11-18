"""Build the checked-in authored scene roster; never fetch images or prompts."""
import hashlib
import json
from pathlib import Path
import random

# Eight distinct scene subjects per thematic family. Families never cross splits.
# These are authored prompts, not claims about independent generated pixel samples.
FAMILIES = {
"alpine": "glacier crevasse|mountain rescue hut|snow-covered cable car|ibex on a ledge|avalanche barrier|frozen summit lake|wooden trail sign|ski patrol sled",
"reef": "brain coral colony|sea turtle feeding|clownfish anemone|sunken anchor|giant clam|sea fan forest|cleaner shrimp|reef research buoy",
"bakery": "sourdough cooling rack|flour-dusted kneading bench|croissant proofing tray|stone bread oven|pastry display case|baker weighing rye|sugar sculpture|stacked bread baskets",
"desert": "wind-carved sandstone arch|camel caravan|salt-flat mirage|abandoned water pump|cactus flowering|canvas expedition camp|dune buggy tracks|oasis irrigation channel",
"observatory": "radio telescope dish|refractor dome|meteorite display|astronomer control desk|solar projection screen|satellite tracking antenna|star-chart archive|telescope mirror workshop",
"harbor": "tugboat towing|container crane|pilot boarding ladder|fishing net repair|dockside fuel hoses|harbor lighthouse|shipyard dry dock|stacked cargo pallets",
"orchard": "apple harvest ladder|pear packing shed|cherry blossom aisle|peach sorting belt|pruning tools|beehives under plum trees|apricot irrigation ditch|fruit farmer tractor",
"pottery": "spinning pottery wheel|glaze mixing bowls|wood-fired kiln|clay extrusion press|ceramic drying shelves|cracked raku vase|decorative tile mural|potter trimming a bowl",
"rainforest": "leafcutter ant trail|tree frog on moss|liana canopy bridge|buttress-root tree|orchid-covered branch|forest research shelter|river otter den|fallen log fungi",
"railway": "steam locomotive depot|signal lever frame|sleeping-car corridor|rail maintenance trolley|station clock tower|freight wagon coupler|trackside semaphore|turntable inspection pit",
"icecoast": "penguin nesting colony|icebreaker bow|polar research tent|blue iceberg tunnel|seal breathing hole|snowmobile supply cache|coastal weather mast|ice shelf fracture",
"bookbinding": "leather embossing press|stitched manuscript spine|marbled paper bath|gold-leaf lettering|paper cutting guillotine|book restoration cradle|linen thread spools|archival map drawer",
"wetland": "heron stalking reeds|boardwalk observation hide|water-lily pond|beaver dam|reed harvesting boat|marsh fire lookout|cattail seed heads|water vole burrow",
"glassworks": "molten glass furnace|glassblower shaping a globe|stained-glass assembly|annealing oven trays|glass bead torch|crystal engraving lathe|recycled bottle cullet|blown-glass chandelier",
"market": "spice merchant stall|flower market cart|basket of root vegetables|antique brassware stand|street noodle counter|woven rug display|fishmonger ice table|open-air tea seller",
"caves": "limestone stalactites|underground river crossing|caver rope descent|glowworm chamber|crystal cave wall|lava tube entrance|archaeological cave trench|subterranean lake raft",
"aviation": "glider tow launch|helicopter maintenance bay|airport windsock|aircraft tire stack|cockpit instrument trainer|seaplane pontoon dock|biplane wing repair|radar control room",
"textile": "handloom weaving|indigo dye vats|wool carding machine|silk cocoon sorting|lace-making pillow|fabric pattern cutting|embroidery hoop|cloth printing blocks",
"savanna": "giraffe browsing acacia|termite mound|lion under a bush|zebra waterhole|elephant dust bath|grassland ranger jeep|ostrich nest|wildebeest riverbank",
"musicworkshop": "violin carving bench|piano action repair|brass trumpet polishing|drum skin stretching|guitar fret installation|harp string winding|pipe-organ tuning|wooden flute drilling",
"canals": "lock gate mechanism|narrowboat cabin|canal towpath bridge|water-level gauge|barge rope coil|aqueduct crossing|canal dredging vessel|lock keeper cottage",
"laboratory": "centrifuge rotor|microscope slide tray|sealed glovebox|precision balance|glass distillation column|spectrometer console|pipette calibration bench|vacuum chamber",
"volcano": "basalt lava columns|sulfur vent|lava-flow monitoring camera|volcanic ash sampling|crater rim shelter|pumice beach|geothermal mud pool|cooled lava rope texture",
"theater": "stage rigging ropes|costume wardrobe|painted backdrop workshop|spotlight catwalk|puppet control frame|empty orchestra pit|prop sword rack|velvet auditorium seats",
"rivercraft": "wooden canoe carving|ferry crossing|kayaker in rapids|floating water mill|river salvage crane|rowing shell boathouse|paddle steamer wheel|riverbank boat slipway",
"gardens": "bonsai pruning table|formal hedge maze|stone meditation garden|greenhouse fern collection|rose pergola|vegetable seedling bench|rain garden swale|garden sundial",
"archaeology": "Roman mosaic excavation|buried pottery shard grid|ancient stone lintel|field artifact washing|dig-site measuring frame|bronze tool conservation|ruined bathhouse|excavation soil profiles",
"bicycle": "wheel truing stand|bicycle courier rack|tandem frame workshop|velodrome starting line|puncture repair tools|cargo bicycle loading|chain lubrication station|mountain bike suspension service",
"coastalbirds": "puffin cliff burrow|gannet diving flock|pelican fishing pier|tern nesting sandbar|oystercatcher shoreline|cormorant drying wings|albatross windswept ledge|sandpiper tidal pool",
"clockmaking": "watchmaker loupe|brass clock escapement|pendulum weight case|gear cutting machine|marine chronometer|cuckoo clock carving|pocket watch restoration|tower clock hand assembly",
"vineyard": "grape harvesting shears|wine barrel cellar|vine trellis repair|grape pressing vat|bottling conveyor|vineyard frost fan|tasting room glassware|terraced grape rows",
"urbanwater": "rooftop water tank|storm drain inspection|fountain pump room|reservoir spillway|water-treatment settling basin|municipal pipe repair|rainwater collection barrel|public drinking fountain",
"spacecraft": "lunar rover prototype|spacesuit glove test|rocket engine nozzle|mission control screen wall|satellite solar panel|orbital capsule mockup|thermal vacuum test rig|parachute recovery package",
"carpentry": "hand plane shavings|timber mortise joint|cabinet clamping frame|sawmill log carriage|wooden stair assembly|carved door panel|veneer pressing bench|joiner tool chest",
"tundra": "musk ox herd|lichen-covered stones|permafrost sampling core|arctic fox shelter|dwarf willow patch|reindeer migration trail|tundra research boardwalk|seasonal thaw pond",
"printing": "letterpress type drawer|offset printing cylinder|screen-print drying rack|lithography stone|newspaper folding line|ink mixing knife|woodcut printing block|book page collation table",
"agriculture": "combine unloading grain|rice paddy transplanting|seed drill mechanism|hay baling field|farm irrigation reel|potato harvesting conveyor|compost turning machine|green manure cover crop",
"blacksmith": "anvil and glowing billet|forge bellows|horseshoe shaping|iron gate scrollwork|quenching trough|hammer polishing station|wrought-iron railing|coal forge chimney",
"freshwater": "trout spawning gravel|dragonfly on rushes|freshwater mussel bed|submerged pond weeds|kingfisher perch|stream ecology sampling|water strider ripples|crayfish under stones",
"ceremony": "lantern festival float|paper carnival mask|ceremonial drum circle|hand-painted parade banner|floral procession arch|community ribbon dance|festival kite launch|harvest wreath display",
"mining": "copper mine conveyor|underground ore cart|mineral sample cabinet|open-pit drilling rig|coal washery screen|mine ventilation fan|safety lamp charging rack|tailings monitoring station",
"hospital": "empty rehabilitation gym|sterilization instrument trays|ambulance loading bay|ultrasound training room|prosthetic fitting workshop|hospital pharmacy shelves|wheelchair repair station|clinical simulation mannequin",
"islands": "palm-thatched pier|island rain cistern|outrigger canoe beach|coastal volcanic chapel|tropical island airstrip|coral-stone village wall|coconut husking station|lagoon navigation poles",
"basketry": "willow soaking trough|rattan chair weaving|coiled grass basket|reed mat loom|bamboo splitting knife|woven fish trap|basket handle repair|wicker storage chest",
"sportsfield": "pole vault setup|archery target range|rowing starting pontoon|rugby goalposts|baseball batting cage|cricket pitch roller|javelin marking line|track hurdles stack",
"photography": "large-format bellows camera|darkroom enlarger|film drying cabinet|contact sheet inspection|studio softbox setup|camera repair bench|cyanotype exposure frame|portrait backdrop stand",
"forestry": "portable sawmill|forest firebreak|tree nursery trays|timber cable skidder|sap collection buckets|wildlife camera trap|woodland footbridge|forest canopy crane",
"jewelry": "gemstone faceting wheel|silver soldering bench|pearl stringing tray|lost-wax casting flask|gold wire drawing plate|ring sizing mandrel|enamel jewelry kiln|lapidary polishing drum",
"nightcity": "late-night tram stop|neon laundromat|illuminated pedestrian tunnel|night street sweeper|rooftop jazz terrace|rainy taxi queue|midnight noodle kitchen|lit riverside footpath",
"beekeeping": "hive frame inspection|honey extraction centrifuge|beeswax candle mold|smoker and hive tools|queen-rearing box|pollinator flower strip|honeycomb capping knife|apiary winter insulation",
"dairying": "milking parlor|cheese aging shelves|butter churning barrel|milk cooling tank|yogurt filling line|dairy laboratory sampler|cream separator|alpine dairy curd vats",
"metalscience": "electron microscope chamber|tensile testing machine|polished metal specimen|heat treatment furnace|hardness indentation rig|metal powder sieve|fatigue testing fixture|weld cross-section display",
"sailing": "sailmaker sewing table|yacht winch rigging|compass navigation station|sailing dinghy launch|spinnaker drying lawn|mast stepping crane|reefed sail in wind|liferaft inspection",
"botanical": "herbarium specimen cabinet|seed bank cold room|plant tissue culture jars|botanical illustration desk|pollen microscopy station|root observation chamber|rare cactus greenhouse|aquatic plant tanks",
"rescue": "mountain stretcher drill|lifeguard rescue board|fire engine hose reel|search dog training course|rope rescue tripod|flood rescue inflatable|emergency radio station|avalanche beacon practice",
"culinary": "chef sharpening knife|handmade pasta drying|sushi rice preparation|clay tandoor cooking|dumpling folding table|open charcoal grill|chocolate tempering slab|copper jam kettle",
"weather": "weather balloon release|anemometer tower|hail measurement board|snow depth stakes|rain gauge station|cloud radar antenna|ocean weather buoy|sunshine recorder sphere",
"leatherwork": "saddle stitching bench|leather skiving tool|boot lasting mold|belt buckle assembly|leather dye drying rack|saddler tool rolls|handbag pattern pieces|embossed leather journal",
"desertwildlife": "fennec fox tracks|scorpion under ultraviolet light|horned lizard camouflage|jerboa burrow entrance|desert tortoise shade|sandgrouse watering|beetle on a dune|sidewinder snake trail",
"marinemapping": "hydrographic survey launch|sonar transducer repair|bathymetric map table|ocean sediment corer|tide recorder hut|marine navigation chart|underwater survey robot|seafloor sampling crane",
"wintersport": "curling stone preparation|ice hockey skate sharpening|biathlon shooting lane|snowboard waxing bench|luge start ramp|ski jump landing slope|speed skating oval|cross-country trail groomer",
"papercraft": "origami crane installation|paper lantern assembly|handmade paper vat|quilling artwork|paper-cut window decoration|cardboard architectural model|papier-mache animal|pop-up book mechanism",
"renewable": "wind turbine nacelle|solar panel cleaning robot|tidal turbine blade|biogas digester|micro-hydro generator|solar thermal mirrors|battery storage room|wave-energy test buoy",
"paleontology": "dinosaur fossil preparation|ammonite collection|fossilized fern slab|plaster field jacket|prehistoric footprint cast|museum skeleton mount|fossil shark tooth sieve|petrified wood display",
"community": "community repair cafe|public chess tables|neighborhood seed library|volunteer soup kitchen|shared allotment shed|community radio booth|public mural painting|village notice board",
"mechanical": "gearbox cutaway|hydraulic cylinder test|industrial bearing assembly|pneumatic valve bench|machine lathe chuck|precision milling table|belt-driven compressor|robot gripper calibration",
"waterfalls": "horseshoe cascade|forest plunge pool|waterfall viewing bridge|basalt curtain falls|mist-covered ravine|stepped travertine pools|waterfall erosion channel|cascading mountain stream",
"perfume": "flower distillation still|perfumer scent organ|essential oil amber bottles|resin grinding mortar|fragrance blotter strips|botanical maceration jars|citrus oil press|perfume bottle engraving",
"transportmuseum": "vintage tram interior|historic fire wagon|wooden railway coach|early electric vehicle|postal delivery cart|restored city bus|steam road roller|antique motorcycle exhibit",
"geology": "folded sedimentary cliff|granite quarry face|geologist compass station|river pebble classification|mineral thin-section microscope|fault-line outcrop|sandstone cross-bedding|geological core storage",
"aquarium": "jellyfish circular tank|seahorse nursery|kelp aquarium tunnel|aquarist feeding platform|coral propagation table|ray touch-pool barrier|filtration pump gallery|quarantine fish tanks",
"ceramicarchitecture": "glazed brick facade|decorative roof tiles|mosaic courtyard floor|terracotta ventilation screen|porcelain wall panels|earthen chimney pots|ceramic street number plaque|tiled public staircase",
"animalcare": "equine rehabilitation pool|dog grooming station|bird rehabilitation aviary|veterinary instrument cart|animal shelter enrichment yard|cat climbing enclosure|hoof trimming stocks|wildlife release crate",
"navigation": "sextant calibration desk|compass rose pavement|lighthouse lens workshop|signal flag locker|ship wheelhouse|navigation beacon service|nautical knot teaching board|marine chronometer cabinet",
"toymaking": "painted locomotive toy kit|teddy bear stitching|tin toy winding mechanism|dollhouse furniture bench|marionette carving|model airplane balsa frame|spinning top lathe|toy block painting",
"earthworks": "stone retaining wall|road grader blade|bridge foundation caisson|excavator trench work|rammed-earth wall form|tunnel boring cutterhead|drainage culvert installation|levee reinforcement bags",
"insects": "mantis on a branch|butterfly emergence case|stag beetle on bark|damselfly wing detail|silkmoth cocoon|hoverfly on flowers|stick insect camouflage|cicada shell on a trunk",
"archives": "historical photograph boxes|ancient parchment flattening weights|wax seal collection|archival film reels|document humidity chamber|scroll storage racks|microfilm reading desk|catalog card cabinet",
"underwaterwork": "diver welding practice|subsea cable spool|underwater pipeline inspection|diving bell platform|remotely operated vehicle|marine propeller cleaning|underwater archaeological grid|diver communication helmet",
"lightingdesign": "handmade paper lampshade|stage gobo projection|architectural wall washing|fiber-optic ceiling|prismatic window experiment|sculptural floor lamp|museum object spotlight|outdoor pathway lanterns",
}


def digest(domain, value):
    return hashlib.sha256(json.dumps(["sr-colab-roster-v1", domain, value], sort_keys=True).encode()).hexdigest()


def build():
    assert len(FAMILIES) == 80
    families = sorted(FAMILIES)
    random.Random(2026111801).shuffle(families)
    split_sizes = [("train", 32), ("dev", 8), ("model_selection", 8), ("calibration", 16), ("test", 16)]
    rows, start = [], 0
    for split, count in split_sizes:
        for family in families[start:start+count]:
            scenes = FAMILIES[family].split("|")
            assert len(scenes) == 8
            for index, scene in enumerate(scenes):
                slot = len(rows)
                prompt = f"A documentary photograph of {scene}, with the surrounding environment clearly visible. Natural materials, realistic proportions, rich spatial detail, balanced light."
                rows.append({"source_id": f"scene-{slot:04d}", "source_group": family,
                             "template_family": family, "split": split, "prompt": prompt,
                             "generation_seed": int(digest("generation-seed", slot)[:8], 16),
                             "key_index": int(digest("key-assignment", slot)[:8], 16) % 16,
                             "reference_nonce": digest("reference-nonce", slot)})
        start += count
    assert len(rows) == len({r["prompt"] for r in rows}) == 640
    return {"schema": "sr-authored-scenes-v1", "note": "640 distinct authored scenes; thematic families split-isolated; not evidence of pixel-population independence", "sources": rows,
            "keys": {split: [digest("writing-key:"+split, i) for i in range(16)] for split, _ in split_sizes}}


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1]/"configs/self_reference_v1_sources.json"
    path.write_text(json.dumps(build(), indent=2)+"\n")
    print(path)
