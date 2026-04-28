import re
import unicodedata
import pandas as pd
from difflib import get_close_matches
from itertools import combinations
from collections import Counter

# ---------------------------------------------------------------------------
# Whitelists: canonical name -> list of normalized aliases (accent-less,
# lowercase). The canonical value is the form that appears in the final graph.
# ---------------------------------------------------------------------------

PEOPLE_CANONICAL = {
    "Tejero":            ["tejero", "coronel tejero", "teniente coronel tejero",
                          "don antonio tejero", "antonio tejero", "tcol tejero"],
    "Alfonso Armada":    ["armada", "general armada", "alfonso armada", "armada comyn"],
    "Milans del Bosch":  ["milans", "milans del bosch", "milans del bosc",
                          "milanos", "milan del bosch", "jaime milans"],
    "Suárez":            ["suarez", "adolfo suarez", "presidente suarez",
                          "presidente del consejo", "presidente del consejo de ministros",
                          "presidente consejo ministros", "presidente del gobierno"],
    "Rey Juan Carlos":   ["juan carlos", "rey juan carlos", "su majestad", "el rey",
                          "zarzuela", "majestad", "jefe del estado",
                          "rey don juan", "rey don juan carlos",
                          "juan gris", "juanito"],  
    "Gutiérrez Mellado": ["gutierrez mellado", "mellado", "manuel gutierrez mellado",
                          "vicepresidente primero", "vicepresidente", "primer vicepresidente",
                          "vicepresidente del gobierno"],
    "Carrillo":          ["carrillo", "santiago carrillo"],
    "Calvo Sotelo":      ["calvo sotelo", "leopoldo calvo sotelo"],
    "Cortina":           ["cortina", "jose luis cortina", "comandante cortina",
                          "cortina prieto"],
    "Pardo Zancada":     ["pardo zancada", "pardo", "comandante fardo"],
    "García Carrés":     ["garcia carres", "juan garcia carres",
                          "carres", "sr. carres", "sr carres","juanillo"],
    "Sabino Fernández Campo": ["sabino", "fernandez campo"],
    "Ibáñez Freire":     ["ibanez freire", "freire", "coronel ibanez", "coronel ibanez ingles",
                          "ibanez ingles", "tcol ibanez", "coro nel ibanez"],
    "Sánchez Valiente":  ["sanchez valiente", "capitan sanchez valiente",
                          "capitan gil sanchez valiente", "valiente portillo",
                          "san chez-valiente", "capitan gil sanchez valiente portillo",
                          "gil sanchez valiente"],
    "Abad Gutiérrez":    ["abad", "capitan abad", "abad gutierrez",
                          "jose luis abad gutierrez"],
    "Bobis González":    ["bobis", "bobis gonzalez", "capitan bobis",
                          "capitan bobis gonzalez", "enrique bobis gonzalez"],
    "Batista":           ["batista", "capitan batista", "batista munecas",
                          "batista muñecas"],
    "Escandell":         ["escandell", "coronel escandell", "coronell escandell",
                          "sr. escandell", "sr escandell"],
    "Aramburu":          ["aramburu", "general aramburu", "aramburu topete",
                          "teniente general aramburu", "ceneral aramburu",
                          "jose aramburu topete", "topete"],
    "Torres Rojas":      ["torres rojas", "general torres rojas",
                          "gral. torres rojas", "gral torres rojas",
                          "general torres"],
    "Gabeiras":          ["gabeiras", "general gabeiras", "gral. gabeiras",
                          "gral gabeiras", "jose gabeiras montero"],
    "Juste":             ["juste", "general juste", "general juste fernandez",
                          "gral. juste", "gral juste"],
    "Manchado":          ["manchado", "coronel manchado", "manchado garcia",
                          "miguel manchado"],
    "Boza":              ["boza", "boza carranco", "boza carranca",
                          "teniente boza", "teniente boza carranco"],
    "Lázaro Corthay":    ["lazaro", "lazaro corthay", "capitan lazaro corthay",
                          "carlos lazaro corthay"],
    "Caruana":           ["caruana", "gral. caruana", "gral caruana"],
    "Muñoz Perea":       ["munoz perea", "sr. munoz perea", "sr munoz perea",
                          "antonio munoz perea"],
    "Gómez Iglesias":    ["gomez iglesias", "capitan gomez iglesias",
                          "vicente gomez iglesias", "cap gomez iglesias"],
    "Franco":            ["franco", "dictador franco", "general franco",
                          "francisco franco"],
    "Felipe González":   ["felipe gonzalez", "don felipe gonzalez"],
    "Prieto":            ["general prieto", "prieto"],
    "Hermosilla":        ["hermosilla", "sr. hermosilla", "sr hermosilla"],
    "López Montero":     ["lopez montero", "sr. lopez montero", "sr lopez montero",
                          "angel lopez-montero", "lopez mon tero", "lopez/ silva"],
    "Laína":             ["laina", "francisco laina", "sr. laina", "sr laina",
                          "senor laina", "señor laina", "sr.laina", "francisco leina"],
    "Sáenz de Santamaría": ["saenz de santamaria", "santamaria", "jose saenz de santamaria",
                             "saenz santamaria", "gral. santamaria"],
    "Coronel San Martín":["coronel san martin", "coronel sanmartin", "coronel san martín"],
    "Pilar Urbano":      ["pilar urbano"],
    "Alcalá Galiano":    ["alcala galiano", "coronel alcala galiano",
                          "coronel alcala-galiano", "felix alcala-galiano",
                          "felix alcald- galiano"],
    "Muñecas":           ["munecas", "capitan munecas", "munecas aguilar",
                          "vicente munecas", "jesus munecas", "capitan vicente munecas",
                          "muñecas aguilar"],
    "Pascual Gálvez":    ["pascual galvez", "capitan pascual galvez",
                          "capitan pascual", "jesus pascual galvez",
                          "pascual galvez cap"],
    "Camacho":           ["camacho", "capitan camacho", "capitan pascual camacho",
                          "jose camacho", "diego camacho"],
    "Álvarez Arenas":    ["alvarez arenas", "capitan alvarez arenas",
                          "capitan alvarez-arenas", "carlos alvarez arenas"],
    "Cid Fortea":        ["cid fortea", "capitan cid", "capitan cid fortea",
                          "jose cid fortea", "jose cid"],
    "García Almenta":    ["garcia almenta", "capitan garcia almenta",
                          "garcia almenta dobon", "fran garcia almenta"],
    "Izquierdo":         ["izquierdo", "teniente izquierdo", "jesus izquierdo",
                          "izquierdo sanchez"],
    "Monge Segura":      ["monge segura", "sr. segura", "sr segura",
                          "rafael monge", "monge"],
    "Quintana":          ["quintana", "sr. quintana", "quintana aparicio",
                          "sr. quintana aparicio", "juan quintana",
                          "quintana lacacci"],
    "Villalonga":        ["villalonga", "sr. villalonga", "garcia villalonga",
                          "sr. garcia villalonga", "sr./ villalonga"],
    "Esquivias":         ["esquivias", "general esquivias", "gral. esquivias",
                          "sr. esquivel", "gral~ esquivias",
                          "ayudante del general esquivias"],
    "Novalvos":          ["novalvos", "sr. novalvos", "manuel novalvos",
                          "novalwos", "novalbos", "sr. manuel novalvos",
                          "jos~ miguel novarbos"],
    "Zugasti":           ["zugasti", "sr. zugasti", "jose zugasti",
                          "zugasti pellejero", "jose zugasti pellejero",
                          "sr. zugasti pellejero", "sr. zugásti"],
    "Bonel":             ["bonel", "comandante bonel", "bonell", "tcol bonelli",
                          "emilio bonell", "bonell esperanza",
                          "comandante bonell esperanza"],
    "Pérez-Llorca":      ["perez-llorca", "perez llorca", "jose pedro perez-llorca",
                          "jose pedro perez llorca", "jose pedro perez"],
    "Oliart":            ["oliart", "sr. oliart", "alberto oliart"],
    "Rodríguez Sahagún": ["rodriguez sahagun", "rodriguez sahagún"],
    "Fraga":             ["fraga", "sr. fraga", "manuel fraga"],
    "Areilza":           ["areilza", "sr. areilza", "jose maria areilza"],
    "Blas Piñar":        ["blas pinar", "blas piñar"],
    "Alexander Haig":    ["alexander haig", "haig", "secretario haig"],
    "Liñán":             ["linan", "sr. linan", "sr linan", "señor linan",
                          "sr. liñan", "sr.· linan"],
    "Martín Fernández":  ["martin fernandez", "sr. martin fernandez",
                          "pedro martin fernandez", "letrado sr. martin fernandez"],
    "Acera":             ["acera", "capitan acera", "teniente acera",
                          "acera aznarez", "manuel acera"],
    "Pascual Galmés":    ["pascual galmes", "general pascual galmes",
                          "capitan general pascual galmes",
                          "teniente general pascual galmes",
                          "antonio pascual galmes"],
    "Miguel Ángel Aguilar": ["miguel angel aguilar", "miguel angel agular",
                             "miguel angel aguillar", "angel aguilar"],
    "Luis Álvarez Rodríguez": ["luis alvarez rodriguez", "alvarez rodriguez",
                               "general alvarez rodriguez", "luis alvarez"],
    "Sánchez Covisa":    ["sanchez covisa", "sr. sanchez covisa"],
    "Ramos Rueda":       ["ramos rueda", "vicente ramos rueda"],
    "Sales Maroto":      ["sales maroto", "sargento sales maroto",
                          "miguel sales maroto"],
    "Ortiz Ortiz":       ["ortiz ortiz", "julio ortiz ortiz", "sr. ortiz ortiz"],
    "Ignacio Román":      ["ignacio roman", "capitan ignacio roman",
                          "francisco ignacio roman"],
    "Nieto":             ["nieto", "sr. nieto", "nieto funcia"],
    "Alonso Hernáiz":    ["alonso hernaiz", "alonso herraez", "alonso herraiz"],
    "Calvo Serer":       ["calvo serer"],
    "Martín Villa":      ["martin villa", "martin-villa"],
    "Tierno Galván":     ["tierno galvan", "enrique tierno galvan"],
    "Fontán":            ["fontan"],
    "Landelino Lavilla": ["landelino lavilla", "sr. lavilla"],
    "Jordi Pujol":       ["jordi Pujol", "jordi pujol"],
    "Alfonso Guerra":    ["alfonso guerra"],
    "Sainz Rodríguez":   ["sainz rodriguez"],
    "Múgica Herzog":     ["mugica herzog", "sr. mugica", "mugica"],
    "Solé Tura":         ["sole tura", "sr. sole tura"],
    "Roca":              ["roca"],
    "Blancafort":       ["blancafort", "javier blancafort"],
    "Capitán Caballero": ["capitan caballero", "caballero"],
    "Alcalde":           ["alcalde", "sr. alcalde", "alcalde de madrid"],
    "Vicente Carricondo": ["vicente carricondo", "carricondo"],
    "Camilo Menéndez":   ["camilo menendez", "general camilo menendez",
                          "almirante camilo menendez"],
    "Manuel Cervantes Rosell": ["manuel cervantes rosell", "cervantes rosell"],
    "Comandante Centeno": ["comandante centeno", "centeno"],
    "Capitán Acera":     ["capitan acera", "capitanes acera"], 
    "Gómez García":      ["gomez garcia", "sr. gomez garcia"],
    "Sanz Arribas":      ["sanz arribas", "sr. Sanz Arribas", "sr. Sanz"],
    "Villar Arregui":    ["villar arregui", "manuel villar arregui"],
    "Iglesias Llamazares": ["iglesias llamazares", "iglesias llameal",
                           "iglesias llamaal"],
    "Labernia Marco":    ["labernia marco", "sr. labernia marco"],
    "Sanz López":        ["sanz lopez", "sr. sanz lopez"],
    "Adolfo Salvador":   ["adolfo salvador", "general adolfo salvador"],
    "Menéndez":          ["menendez", "sr. menendez"], 
    "José Salva Paradela": ["jose salva paradigmela", "salva paradigmela"],
    "José Corral Rodríguez": ["jos~ corral rodriguez", "jose corral rodriguez"],
    "José Núñez":        ["jose nuñez"],
    "Navío López Rolandi": ["navio lopez rolandi", "navio lopez", "lopez rolandi"],
    "López Rolandi":     ["lopez rolandi"],
    "Princesa Sofía":      ["princesa sofia", "doña sofia", "sofia de marichalar"],
    "Gustavo Urrutia":     ["gustavo urrutia", "sr. urrutia"],
    "Nicolás García":      ["nicolas garcia", "sr. nicolas garcia"],
    "Pepe Cassinello":    ["pepe cassinello", "cassinello"],
    "General Lluch":       ["gral. lluch", "general lluch", "lluch"],
    "César Álvarez Fernández": ["alvarez fernandez", "cesar alvarez"],
    "Pedro Mas Oliver":   ["pedro mas oliver", "pedro mas olliver", "mas oliver"],
    "José Arregui":      ["jose arregui", "arregui"],
    "Ángel Martínez Juan": ["angel martinez juan", "amgel martinez juan"],
    "General Urrutia":   ["general urrutia", "urrutia"],
    "Guillermo Ostos":   ["guillormo ostos", "sr. ostos", "ostos mateo-cañero"],
    "Coronel Monzón":    ["coronel monzon", "monzon"],
    "Coronel Santos":    ["coronel santos", "santos"],
    "Capitán Merlo":     ["capitan merlo", "merlo"],
    "Félix Porras Blanco": ["felix porras blanco", "porras blanco"],
    "Coronel Valencia Remón": ["coronel valencia remon", "valencia remon"],
    "General Toquero":   ["gral. toquero", "general toquero", "toquero"],
    "José Moreno Wirtz": ["jose moreno wirtz", "jos~ moreno wirtz", "moreno wirtz"],
    "Almirante Carrero":    ["almirante carrero", "almirante carrerro"],
    "Guillermo Estévez Boero": ["guillermo estrevez boero", "dr. guillermo estrevez boero", "dr. estrevez boero"],
    "Quintana Sanjuán":   ["quitero sanjuan", "quintana sanjuan", "sr. quintero sanjuan"],
    "Sargento Rando":     ["sargento rando"],
    "Alvarez Sola":        ["alvarez sola"],
    "Del Pozo Pérez":      ["del pozo perez", "sr. del pozo perez"],
    "General León Pizarro": ["general leon pizarro", "gral. leon pizarro"],
    "Julián Marías": ["julian marias"],
    "Juan Montero Ramírez": ["juan montero ramirez"],
    "José Vázquez García": ["jos~ vazquez garcio", "jose vazquez garcia"],
    "Teniente Coronel Gibert": ["tcol gibert"],
    "Comandante Goróstegui": ["comandante goréstegui"],
    "General Ballesteros": ["ballesteros", "general ballesteros"],
    "José Faura": ["jose faura", "general faura"],
    "Carmela García Moreno": ["carmela garcia moreno", "sra. carmela garcia moreno"],
    "Capitán Dusmet": ["capitan dusmet", "francisco dusmet", "dusmet"],
    "Teniente Ramos": ["teniente ramos", "teniente ramos.-", "vicente ramos"],
    "Teniente Núñez Ruano": ["teniente nuñez ruano", "nuñez ruano"],
    "José Antonio Assiego": ["jose antonio assiego"],
    "Tent Soler": ["sr. tent soler"],
    "Martínez García": ["martinez garcia"],
    "Luis Torres": ["luis torres"],
    "Juan Plá": ["sr. pla"],
    "Salvá Paradela": ["sr. salva"],
    "Enrique Pérez Hernández": ["enriqué pérez hernández"],
    "Castillo Ortega": ["castillo ortega"],
    "Capitán Mochales": ["capitan mochales", "mochales"],
    "General Pérez Íñigo": ["general perez inigo"],
    "Juan Antonio Tévar Gómez": ["juan antonio tevar gomez"],
    "Fernando Sanz Esteban": ["fernando sanz esteban", "sanz esteban", "fernando sanz esteban.- regimien"],
    "Antonio Pérez Crespo": ["antonio perez crespo"],
    "Juan Fone Ferrar": ["juan fone ferrar"],
    "Gómez González": ["gomez gonzalez"],
    "Subteniente Presa": ["subteniente presa"],
    "Cadalso Preciados": ["cadalso preciados"],
    "Juan Antequera Betrán": ["juan antequera betran"],
    "Capitán Pérez": ["capitan perez", "capitan pérez"],
    "Fernando Arias": ["fernando arias"],
    "General Ortiz Call": ["fernando ortiz call", "general ortiz call"], # Consolidado Final OCR
    "Presidente Reagan": ["presidente reagan"], # Rescatado final
    "Pérez Heredia": ["pérez heredia", "perez heredia"], # Rescatado final
    "Juan": ["juan"], # Nombre ambiguo o clave
}

ORG_CANONICAL = {
    "Guardia Civil": ["guardia civil", "g. civil", "guardias civiles", "benemérita", "guerdia civil"],
    "Ejército":             ["ejercito", "ejercito de tierra", "ejercito espanol", "fuerzas armadas"],
    "CESID":                ["cesid", "seced"],
    "Policía Nacional":     ["policia nacional", "policia", "cuerpo nacional de policia", "fuerza publica", "fuerza pública"],
    "PSOE":                 ["psoe", "partido socialista"],
    "UCD":                  ["ucd", "union de centro democratico"],
    "PCE":                  ["pce", "partido comunista"],
    "ETA":                  ["eta"],
    "Congreso":             ["congreso", "congreso de los diputados", "parlamento"],
    "Ministerio del Interior": ["interior", "ministerio del interior", "ministerio interior"],
    "Televisión Española":  ["tve", "television espanola", "television española", "radio television espanola", "rtve"],
    "Armada Española":      ["armada espanola", "la armada", "marina"],
    "Radio Nacional":       ["radio nacional", "radio nacional de espana", "radio nacional de españa", "cadena ser", "radio cadena"],
    "Fuerza Nueva":         ["fuerza nueva"],
    "División Azul":        ["division azul", "division azul española"],
    "Herri Batasuna":       ["herri batasuna", "hb"],
    "Consejo Supremo":      ["consejo supremo", "consejo supremo de justicia militar", "consejo supremo de justicia"],
    "Liga Comunista Revolucionaria": ["liga comunista revolucionaria", "lcr"],
    "OTAN":                 ["otan", "nato", "alianza atlantica", "alianza atlántica"],
    "Europa Press":         ["europa press"],
    "Falange Española":     ["falange española", "falange espanola", "falange española autentica", "falange espanola autentica", "falange española auténtico", "falange española autentico"],
    "Tribunal":             ["tribunal", "tribunal supremo", "tribunal constitucional", "tribunal militar", "tribunal constitucio"],
    "Alfonso Armada":       ["general armada", "gral. armada", "gral armada", "gene ral armada", "general/ armada"],
    "Milans del Bosch":     ["general milans", "general milans del bosch"],
    "Aramburu":             ["general aramburu", "gral. aramburu"],
    "Comunión Tradicionalista": ["comunion tradicionalista", "comunion carlista", "comunion catolico monarquica", "carlismo", "carlistas", "partido carlista"],
    "USO":                  ["uso", "central sindical uso", "union sindical obrera", "centrales sindicales uso"],
    "UMD":                  ["umd", "union militar democratica", "asociacion democratica", "militares democraticos", "militares de la democracia"],
    "División Acorazada Brunete": ["division acorazada brunete", "brunete", "dac brunete", "dac", "division acorazada", "division blindada"],
    "Astilleros Españoles": ["astilleros espanoles", "astilleros", "astillero espanol"],
    "Frente Anticomunista Español": ["frente anticomunista español", "frente ánticomunista español"],
    "Comité Nacional": ["comite nacional"],
    "Unión Militar Española": ["unión militar española", "ume"],
    "Juventudes Comunistas Revolucionarias": ["juventudes comunistas revolucionarias"],
    "Empresa Nacional Bazán": ["bazan", "empresa nacional bazan"],
    "Partido Nacional Independiente": ["partido nacional independiente"],
}


# ---------------------------------------------------------------------------
# Pre-limpieza OCR
# ---------------------------------------------------------------------------

def _clean_ocr(text: str) -> str:
    text = re.sub(r"[~|`´¨]", "", text)
    text = re.sub(r"[—–-]+", " ", text)
    text = re.sub(r"(?<=\w)[.,;:](?=\w)", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\(\)]", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Generic blacklist
# ---------------------------------------------------------------------------

GENERIC_BLACKLIST = {
    "fiscal", "sr. fiscal", "teniente fiscal", "fiscal togado",
    "jefe", "jefes", "adjunto", "secretario", "subsecretario",
    "relator", "juez", "magistrado", "magistrados", "letrado", "letrados",
    "rector", "alcalde", "embajador", "almirante", "almirantes",
    "suboficial", "sargento", "teniente", "coroneles", "generales",
    "capitanes", "tenientes", "subteniente", "coronel", "general",
    "ayudante", "ayudantes", "oficial", "oficiales", "defensores",
    "consejero", "consejeros", "diputado", "diputados", "ministros",
    "militares", "soldado", "periodistas", "nacional",
    "tcol", "gral", "tcol.", "gral.", "tte", "cap", "cte", "cmte",
    "comte", "vicepresidente primero", "vicepresidente",
    "presidente del consejo", "jefe del estado", "jefe de estado",
    "director general", "director de seguridad",
    "solo", "bando", "reina", "monarca", "monarquía", "dios",
    "historia", "propaganda", "noticia", "resolución", "respuesta",
    "asamblea", "informe", "organización", "dirección", "administración",
    "presidencia", "comisiones", "sección", "superioridad", "senado",
    "estado", "gobierno", "república", "régimen", "constitución",
    "partidos", "partidos políticos", "nuestro partido",
    "real decreto", "orden", "proceso", "órdenes", "servicios",
    "unidades", "unidad", "grupos", "compañía", "mando", "mandos",
    "división", "brigada", "brigadas", "servicio",
    "febrero",
    "excmo", "excmos", "ilmo", "ilmos", "sres",
    "autoridad civil", "autoridad gubernativa", "autoridad superior",
    "autoridad militar", "autoridades civiles", "autoridades militares",
    "administracion militar", "administración militar",
    "gobierno militar", "gobiernos militares",
    "gobernador civil", "gobernadores civiles",
    "gobernador militar", "gobernadores militares",
    "capitania general", "capitanía general",
    "region militar", "regiones militares",
    "alerta verde", "alerta roja", "alerta",
    "acuartelamiento", "acuartelamientos",
    "brigada antigolpe",
    "estado de sitio", "estado de excepcion",
    "sectores", "fuentes", "mandos militares",
    "elementos", "ciertos elementos", "algunos elementos",
    "personas", "interesados",
    "autoridad", "prensa", "academia", "universidad",
    "iglesia", "nacion",
    "camara", "comite ejecutivo",
    "direccion general", "dirección general",
    "seguridad del estado",
    "gobierno civil", "inteligencia", "informacion",
    "grupos tacticos", "servicio geografico",
    "justicia militar",
    "abogados", "abogados defensores",
    "los senores consejeros",
    "preguntado", "manifiesta", "declarante",
    "nota informativa", "relacion cesid",
    "escoltas nombramiento", "presidente del",
    "capi", "infor", "ttes", "tenien tes",
    "tenientes coroneles", "teniente alvarez",
    "guardias civiles", "don juan", "doña sofia", "princesa sofia",
    "partido politico", "junta democratica",
    "ciudad real propaganda", "los senores consejeros",
    "coordinacion", "informaciones",
    "operacion aricte", "partidos politicos psa",
    "turno del sr. fiscal", "capitanes",
    "coronel juez", "coronel mas oliver",
    "juez instruc tor", "gobernador civil",
    "comite ejecutivo", "realicen eat", "numero cin",
    "tercera region militar", "primer r.m",
    "orden general", "sala olimpia",
    "frente anticomunista español",
    "jefe oficiales", "jofe", "tenient", "sr. fiscal togado", ".fiscal",
    "carta", "contraprestaciones", "contestar", "conel",
    "acabose", "estuvo/", "insolito", "inmedia",
    "sierraeran", "plaza/ lavapias", "radio intercontinental/",
    "pamplona pamplona convocatoria", "unidades milita",
    "meer", "los senores consejeros", "comite ejecutivo",
    "comite ejec utivo", "academia .de", "bosch",
    "coordinacion", "gobernador civil", "policias",
    "servicio geografico", "informacion", "movimiento computadores",
    "operacion aricte", "partidos politicos psa",
    "juzgado militar especial", "investigacion criminal",
    "insolito", "informaciones", "numero cin", "uni dad",
    "terminadas", "concej al", "caracter general",
    "ayudante del tie", "bendala(div",
    "direccion general para iberoamerica",
    "capitanes generales", "reales ordenanzas",
    "primera r.m", "camara alta",
    "frente anticomunista espanol", "teniente coronc",
    "algunos procesados", "centros oficiales", "oficial general",
    "garcia figueras cap", "san sebastian", "rivera", "dare",
    "denegada", "contestar", "menendez", "mesa", "otero",
    "lUIS TORRES", "lacalle leloup", "orozco masieu",
    "policias", "coordinacion", "los senores consejeros",
    "comite ejecutivo", "capitanes general", "servicio geografico",
    "real", "jefe e.m", "oficiales generales",
    "juzgado especial del", "direccion general fuerzas del gar",
    "consejero del gobier", "tercera region militar",
    "fidel castro ruz presidente", "unidades gac",
    "juzgado militar especial", "investigacion criminal",
    "insolito", "informaciones", "inseguridad",
    "operacion aricte", "partidos politicos psa",
    "frente anticomunista espanol", "camara alta",
    "concej al", "negada", "ignora", "contestar", "as imisno",
    "gobernador civíl", "coordinación", "servicio geográfico",
    "policías", "comité ejecutivo", "los señores consejeros",
    "información", "insólito", "investigación criminal",
    "cámara alta", "asimisno", "concejal", "contéstar",
    "sr. presidente", "habia gobierno", "comité nacional",
    "comite zjecativo", "gobíerno mílitor", "gobier",
    "jos manifestuntes", "reticencias", "cámara",
    "capitanes cenerales", "teniei'ltegeneraj", "juez togado instructor",
    "dirección general fuerzas del gar", "operación aricte",
    "grupo servicios especiales/", "grupos tácticos",
    "sr. juez instructor", "organo del poder/",
    "registro general entrada", "tercera región militar",
    "partidos políticos psa", "vicapresidencia dol ini",
    "lito n.ref", "organo del poder/", "lito n.ref",
    "delicias", "bilbao bilbao manifestación",
    "partidos políticos psa", "registro general entrada",
    "vicapresidencia dol ini", "sr. general div", "organo",
    "jos fascistas", "civil detenido", "capitalo",
    "corte español", "partidos poli", "tácilitarias",
    "sr. hernández griñó que=", "cabltdan sue sr. capitán",
    "corte español", "civil", "partidos poli", "tácilitarias",
    "nacionalistas", "servi cio", "prest",
    "gibert crespoy corral rodriguez", "d.jos~ moya gómez",
}


def _no_accent(text: str) -> str:
    text = _clean_ocr(text)
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    ).lower().strip()

def build_lookup_index(canonical_dict: dict) -> tuple[dict, list]:
    index = {}
    for canonical, aliases in canonical_dict.items():
        canonical_norm = _no_accent(canonical)
        index[canonical_norm] = canonical
        for alias in aliases:
            alias_norm = _no_accent(alias)
            index[alias_norm] = canonical
    return index, list(index.keys())

def resolve_entity(raw: str, lookup_index: dict, candidates: list, threshold: float = 0.78) -> tuple[str, bool]:
    norm = _no_accent(raw)
    if norm in lookup_index:
        return lookup_index[norm], True

    substring_matches = []
    for alias, canonical in lookup_index.items():
        if alias in norm:
            substring_matches.append((len(alias), canonical))
        elif norm in alias and alias.startswith(norm) and len(norm) >= 5:
            substring_matches.append((len(alias), canonical))
            
    if substring_matches:
        substring_matches.sort(reverse=True)
        return substring_matches[0][1], True

    matches = get_close_matches(norm, candidates, n=1, cutoff=threshold)
    if matches:
        return lookup_index[matches[0]], True

    fallback = raw.strip().title()
    return fallback, False

def flatten_entities(df_ner: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df_ner.iterrows():
        for entity_type, col in [("PER", "people"), ("ORG", "organizations")]:
            raw_str = str(row.get(col, ""))
            if raw_str in ("", "nan"):
                continue
            for entity in raw_str.split("|"):
                entity = entity.strip()
                if entity:
                    records.append({
                        "doc_id": row["doc_id"],
                        "source": row["source"],
                        "entity_raw": entity,
                        "entity_type": entity_type,
                    })
    return pd.DataFrame(records)

def apply_normalization(
    df_long: pd.DataFrame,
    people_lookup: dict,
    people_candidates: list,
    org_lookup: dict,
    org_candidates: list,
    threshold: float = 0.78,
) -> pd.DataFrame:
    canonicals = []
    in_whitelists = []
    new_types = []

    for _, row in df_long.iterrows():
        raw = row["entity_raw"]
        current_type = row["entity_type"]

        if current_type == "PER":
            canonical, in_wl = resolve_entity(raw, people_lookup, people_candidates, threshold)
            final_type = "PER"
            if not in_wl:
                alt_canonical, alt_in_wl = resolve_entity(raw, org_lookup, org_candidates, threshold)
                if alt_in_wl:
                    canonical, in_wl, final_type = alt_canonical, True, "ORG"
        else:
            canonical, in_wl = resolve_entity(raw, org_lookup, org_candidates, threshold)
            final_type = "ORG"
            if not in_wl:
                alt_canonical, alt_in_wl = resolve_entity(raw, people_lookup, people_candidates, threshold)
                if alt_in_wl:
                    canonical, in_wl, final_type = alt_canonical, True, "PER"

        canonicals.append(canonical)
        in_whitelists.append(in_wl)
        new_types.append(final_type)  

    df_long = df_long.copy()
    df_long["entity_canonical"] = canonicals
    df_long["in_whitelist"] = in_whitelists
    df_long["entity_type"] = new_types  
    return df_long

def cluster_fallbacks(df_long: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    fallbacks = df_long[~df_long["in_whitelist"]].copy()
    if fallbacks.empty:
        return df_long

    canonical_freq = fallbacks.groupby("entity_canonical")["doc_id"].nunique()
    unique_canonicals = canonical_freq.sort_values(ascending=False).index.tolist()
    cluster_map = {}
    canonical_norms = [(c, _no_accent(c)) for c in unique_canonicals]

    for i, (canon_i, norm_i) in enumerate(canonical_norms):
        if canon_i in cluster_map: continue
        cluster_map[canon_i] = canon_i
        for canon_j, norm_j in canonical_norms[i + 1:]:
            if canon_j in cluster_map: continue
            if norm_j in norm_i:
                cluster_map[canon_j] = canon_i 
            elif norm_i in norm_j:
                cluster_map[canon_i] = canon_j
                cluster_map[canon_j] = canon_j
            else:
                matches = get_close_matches(norm_j, [norm_i], n=1, cutoff=threshold)
                if matches:
                    cluster_map[canon_j] = canon_i

    df_long = df_long.copy()
    mask = ~df_long["in_whitelist"]
    df_long.loc[mask, "entity_canonical"] = (
        df_long.loc[mask, "entity_canonical"].map(cluster_map).fillna(df_long.loc[mask, "entity_canonical"])
    )
    return df_long

def frequency_filter(df_long: pd.DataFrame, min_docs: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    doc_freq = df_long.groupby("entity_canonical")["doc_id"].nunique().rename("doc_count").reset_index()
    df_long = df_long.merge(doc_freq, on="entity_canonical", how="left")
    kept = df_long[df_long["in_whitelist"] | (df_long["doc_count"] >= min_docs)].drop(columns=["doc_count"])
    discarded = df_long[~df_long["in_whitelist"] & (df_long["doc_count"] < min_docs)].drop(columns=["doc_count"])
    return kept, discarded

def generate_edges(df_nodes: pd.DataFrame, output_edges_csv: str) -> pd.DataFrame:
    """
    Generate graph edges based on entity co-occurrence
    within the same document (doc_id).
    """
    print("\n[Graph] Building co-occurrence edges by document...")
    
    # 1. Remove duplicates so repeated mentions in one doc count once.
    df_unique = df_nodes.drop_duplicates(subset=["doc_id", "entity_canonical"])
    
    # 2. Group entities by document
    docs = df_unique.groupby("doc_id")["entity_canonical"].apply(list)
    
    # 3. Generate all pair combinations (edges)
    edges = []
    for entities in docs:
        if len(entities) > 1:
            # Sort so (A,B) is the same as (B,A)
            entities = sorted(entities)
            for pair in combinations(entities, 2):
                edges.append(pair)
                
    # 4. Count edge weight for each connection
    edge_counts = Counter(edges)
    
    # 5. Convert to DataFrame
    df_edges = pd.DataFrame(
        [{"Source": src, "Target": tgt, "Weight": weight} for (src, tgt), weight in edge_counts.items()]
    )
    
    if not df_edges.empty:
        df_edges = df_edges.sort_values(by="Weight", ascending=False)
        df_edges.to_csv(output_edges_csv, index=False)
        print(f"[Graph] Generated {len(df_edges)} edges. Saved to {output_edges_csv}")
        print("\nTop 10 strongest connections:")
        print(df_edges.head(10).to_string(index=False))
    else:
        print("[Graph] Warning: no connections found (documents with multiple entities were not detected).")
        
    return df_edges


def run_normalization(ner_csv: str, output_nodes_csv: str, output_edges_csv: str, min_docs: int = 2, threshold: float = 0.78) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_ner = pd.read_csv(ner_csv)
    people_lookup, people_candidates = build_lookup_index(PEOPLE_CANONICAL)
    org_lookup, org_candidates = build_lookup_index(ORG_CANONICAL)

    df_long = flatten_entities(df_ner)
    df_long = apply_normalization(df_long, people_lookup, people_candidates, org_lookup, org_candidates, threshold)

    # Convert to string to avoid float/NaN artifacts before cleanup
    df_long["entity_canonical"] = df_long["entity_canonical"].astype(str)

    df_long = df_long[
        ~df_long["entity_canonical"].str.lower().str.strip().isin(GENERIC_BLACKLIST)
    ].copy()
    
    df_long = cluster_fallbacks(df_long, threshold=0.80)
    kept, _ = frequency_filter(df_long, min_docs)

    output_cols = ["doc_id", "source", "entity_raw", "entity_canonical", "entity_type"]
    df_nodes = kept[output_cols]
    df_nodes.to_csv(output_nodes_csv, index=False)
    
    # Generate edges
    df_edges = generate_edges(df_nodes, output_edges_csv)

    return df_nodes, df_edges