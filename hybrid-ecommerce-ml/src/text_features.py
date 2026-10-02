import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_synthetic_catalogue():
    """Creates an isolated synthetic e-commerce catalogue of 30 core products for feature augmentation."""
    catalogue = [
        # --- Keyboards & Input Devices (3) ---
        {"Product_ID": "P001", "Title": "High-End Ergonomic Mechanical Keyboard", "Category": "Keyboards", "Base_Price": 149.99, "Description": "tactile switches RGB backlighting wrist rest ergonomics custom keycaps wireless dual mode productivity display mechanical keyboard high engagement"},
        {"Product_ID": "P002", "Title": "Precision Wireless Laser Mouse", "Category": "Mice & Trackballs", "Base_Price": 69.99, "Description": "ergonomic precision laser mouse quick response smooth scrolling productivity office setup lightweight casual buyer low duration"},
        {"Product_ID": "P003", "Title": "Programmable Macro Keypad", "Category": "Keyboards", "Base_Price": 79.00, "Description": "auxiliary macro pad rotary dial shortcuts streamer developer productivity fast action key binding quick browse"},

        # --- Monitors & Displays (3) ---
        {"Product_ID": "P004", "Title": "UltraWide 34-inch Curved Productivity Monitor", "Category": "Monitors", "Base_Price": 599.99, "Description": "ultrawide curved monitor high resolution IPS panel color accurate HDR display multi-window productivity dual input HDMI DisplayPort USB-C hub high page values extreme duration"},
        {"Product_ID": "P005", "Title": "4K Ultra HD Smart Desktop Monitor 27-inch", "Category": "Monitors", "Base_Price": 349.99, "Description": "4K UHD crisp display graphic design color calibration high page value product browsing multi-input medium high duration"},
        {"Product_ID": "P006", "Title": "Portable 15.6-inch Full HD OLED Screen", "Category": "Monitors", "Base_Price": 249.99, "Description": "lightweight second screen dual display mobile productivity travel laptop companion USB-C powered returning visitor"},

        # --- Headphones & Audio (4) ---
        {"Product_ID": "P007", "Title": "Active Noise Cancelling Wireless Headphones", "Category": "Headphones", "Base_Price": 249.99, "Description": "over-ear bluetooth headphones active noise cancellation long battery life high audio fidelity microphone touch controls travel case medium engagement"},
        {"Product_ID": "P008", "Title": "Compact Portable Wireless Bluetooth Speaker", "Category": "Audio", "Base_Price": 49.99, "Description": "entry level compact portable speaker waterproof dustproof casual listening quick purchase travel friendly budget sound high bounce rate exit rate"},
        {"Product_ID": "P009", "Title": "True Wireless Noise-Isolating Earbuds", "Category": "Audio", "Base_Price": 129.99, "Description": "in-ear wireless earbuds active noise isolation charging case low latency quick checkout casual buyer low duration"},
        {"Product_ID": "P010", "Title": "Hi-Fi Studio Reference Monitor Headphones", "Category": "Headphones", "Base_Price": 299.99, "Description": "audiophile studio headphones open back high dynamic range mixing mastering acoustic detail intensive research high product duration"},

        # --- Laptops & Workstations (3) ---
        {"Product_ID": "P011", "Title": "Pro Workstation Laptop 16-inch M3", "Category": "Laptops", "Base_Price": 1899.99, "Description": "high performance workstation laptop fast processor extended duration high page values intensive research developer creative pro display extreme page values"},
        {"Product_ID": "P012", "Title": "Ultra-Slim Business Ultrabook 14-inch", "Category": "Laptops", "Base_Price": 999.99, "Description": "lightweight business laptop extended battery life administrative tasks high page value mobility executive productivity high administrative duration"},
        {"Product_ID": "P013", "Title": "Custom Liquid-Cooled Gaming Desktop PC", "Category": "Computers", "Base_Price": 2199.99, "Description": "flagship gaming PC extreme GPU cooling peak frame rates high cart value high duration session intensive specs evaluation extreme duration high page values"},

        # --- Gaming Peripherals (3) ---
        {"Product_ID": "P014", "Title": "Pro Gaming Mouse with Precision Sensor", "Category": "Gaming", "Base_Price": 79.99, "Description": "ultra lightweight gaming mouse high DPI optical sensor programmable buttons fast response low latency RGB lighting PTFE feet medium product duration"},
        {"Product_ID": "P015", "Title": "Wireless Pro Gaming Controller Pad", "Category": "Gaming", "Base_Price": 69.99, "Description": "ergonomic game pad haptic feedback low exit rate accessory add-on fast checkout high conversion returning visitor"},
        {"Product_ID": "P016", "Title": "Virtual Reality Headset & Motion Controllers", "Category": "Gaming", "Base_Price": 499.99, "Description": "immersive 3D VR headset spatial audio motion tracking interactive product page high engagement research high duration high page value"},

        # --- Accessories, Adapters & Storage (5) ---
        {"Product_ID": "P017", "Title": "10-in-1 USB-C Hub Dongle Adapter", "Category": "USB-C Accessories", "Base_Price": 39.99, "Description": "multiport adapter USB-C pass-through charging HDMI 4K Ethernet SD card reader lightweight accessory entry level quick browse low duration"},
        {"Product_ID": "P018", "Title": "1TB NVMe High-Speed Solid State Drive", "Category": "Storage", "Base_Price": 109.99, "Description": "fast NVMe M.2 PCIe 4.0 expansion storage internal SSD high transfer speed game loading file backup technical specs inquiry medium duration"},
        {"Product_ID": "P019", "Title": "Portable Rugged External SSD 2TB", "Category": "Storage", "Base_Price": 199.99, "Description": "high speed shockproof external drive fast backup USB 3.2 compact travel utility quick decision buyer low exit rate"},
        {"Product_ID": "P020", "Title": "Thunderbolt 4 Docking Station", "Category": "USB-C Accessories", "Base_Price": 279.99, "Description": "enterprise USB-C docking station dual 4K display power delivery desk management administrative duration heavy administrative duration"},
        {"Product_ID": "P021", "Title": "Fast Charge Power Bank 20000mAh", "Category": "Power & Cables", "Base_Price": 49.99, "Description": "high capacity power bank fast charging mobile accessory travel quick purchase budget sound high bounce rate"},

        # --- Webcams & Microphones (3) ---
        {"Product_ID": "P022", "Title": "1080p HD Webcam with Privacy Shutter", "Category": "Webcams", "Base_Price": 59.99, "Description": "full HD webcam noise reducing microphone auto low light correction video call conferencing home office setup clear video casual buyer"},
        {"Product_ID": "P023", "Title": "4K Ultra-HD Streaming Webcam", "Category": "Webcams", "Base_Price": 159.99, "Description": "4K HDR webcam auto framing dual mic crystal clear broadcast streaming remote work video conferencing medium engagement"},
        {"Product_ID": "P024", "Title": "Cardioid USB Studio Condenser Microphone", "Category": "Microphones", "Base_Price": 110.00, "Description": "USB condenser mic gain control zero latency headphone monitoring voiceover podcasting clarity clean sound informational duration"},

        # --- Desk, Lighting & Furniture Setup (4) ---
        {"Product_ID": "P025", "Title": "Adjustable Aluminium Laptop Stand", "Category": "Desk Accessories", "Base_Price": 29.99, "Description": "ergonomic laptop riser aluminium cooling ventilation foldability desk setup posture health entry level low cost accessory high bounce rate"},
        {"Product_ID": "P026", "Title": "Smart Desk Monitor LED Light Bar", "Category": "Desk Accessories", "Base_Price": 69.99, "Description": "auto dimming monitor lamp screen glare reduction ambient lighting desk ergonomics eye care touch controls returning visitor"},
        {"Product_ID": "P027", "Title": "Motorized Electric Standing Desk Frame", "Category": "Furniture", "Base_Price": 399.99, "Description": "dual motor height adjustable sit stand workspace ergonomic setup heavy administrative duration desk upgrade long product duration"},
        {"Product_ID": "P028", "Title": "Ergonomic Mesh High-Back Office Chair", "Category": "Furniture", "Base_Price": 299.99, "Description": "breathable mesh lumbar support adjustable headrest long duration comfort workplace health high duration"},

        # --- Networking & Smart Home (2) ---
        {"Product_ID": "P029", "Title": "Wi-Fi 6 Ultra Dual-Band Router", "Category": "Networking", "Base_Price": 149.99, "Description": "high speed wireless router multi device coverage low latency streaming home networking informational inquiry heavy informational duration"},
        {"Product_ID": "P030", "Title": "Smart Home Voice Assistant Speaker", "Category": "Smart Home", "Base_Price": 59.99, "Description": "voice controlled assistant smart automation fast informational query entry level low cost smart home informational duration"}
    ]
    return pd.DataFrame(catalogue)


def map_session_to_profile(row):
    """
    Dynamic session text builder. 
    Constructs a weighted keyword string based on exact numeric values.
    """
    tokens = []

    # 1. Page Values (High impact on luxury/high-end gear)
    pv = float(row.get('PageValues', 0))
    if pv > 100:
        tokens.extend(["extreme page values", "high page values", "flagship", "high cart value"])
    elif pv > 40:
        tokens.extend(["high page value", "high page values", "4K", "workstation"])
    elif pv > 10:
        tokens.extend(["medium page value", "ergonomic", "productivity"])
    else:
        tokens.extend(["entry level", "casual buyer", "budget sound", "accessory"])

    # 2. Product Related Duration
    pr_dur = float(row.get('ProductRelated_Duration', 0))
    if pr_dur > 5000:
        tokens.extend(["extreme duration", "high duration", "custom liquid cooled gaming desktop PC", "intensive research"])
    elif pr_dur > 2000:
        tokens.extend(["high product duration", "extended duration", "ultrawide curved monitor", "high duration"])
    elif pr_dur > 800:
        tokens.extend(["medium product duration", "tactile switches", "mechanical keyboard", "headphones"])
    elif pr_dur > 300:
        tokens.extend(["medium engagement", "gaming mouse", "webcam"])
    else:
        tokens.extend(["low duration", "quick browse", "quick checkout"])

    # 3. Administrative Duration
    admin_dur = float(row.get('Administrative_Duration', 0))
    if admin_dur > 300:
        tokens.extend(["heavy administrative duration", "administrative duration", "enterprise USB-C docking station", "ultrabook"])
    elif admin_dur > 80:
        tokens.extend(["administrative tasks", "desk management", "office setup"])

    # 4. Informational Duration
    info_dur = float(row.get('Informational_Duration', 0))
    if info_dur > 200:
        tokens.extend(["heavy informational duration", "informational inquiry", "Wi-Fi 6 ultra dual band router"])
    elif info_dur > 50:
        tokens.extend(["informational duration", "smart automation"])

    # 5. Bounce Rate & Exit Rate
    br = float(row.get('BounceRates', 0))
    if br > 0.08:
        tokens.extend(["high bounce rate", "casual listening", "quick browse", "low cost accessory"])

    # 6. Visitor Type
    v_type = str(row.get('VisitorType', ''))
    if v_type == 'Returning_Visitor':
        tokens.extend(["returning visitor", "productivity"])
    elif v_type == 'New_Visitor':
        tokens.extend(["new visitor", "entry level"])

    return " ".join(tokens)


class TFIDFSimilarityFeatureGenerator:
    """Computes cosine similarity between session behavioral dynamic query and product catalogue."""
    def __init__(self):
        self.catalogue_df = get_synthetic_catalogue()
        # sublinear_tf smooths out term frequencies for multi-word matches
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
        self.catalogue_vectors = None

    def fit(self, X=None):
        self.catalogue_vectors = self.vectorizer.fit_transform(self.catalogue_df['Description'])
        return self

    def transform(self, X_df):
        if isinstance(X_df, pd.Series):
            X_df = pd.DataFrame([X_df])
            
        profiles = [map_session_to_profile(row) for _, row in X_df.iterrows()]
        profile_vectors = self.vectorizer.transform(profiles)
        sim_matrix = cosine_similarity(profile_vectors, self.catalogue_vectors)
        
        max_sims = np.max(sim_matrix, axis=1)
        best_indices = np.argmax(sim_matrix, axis=1)
        
        matched_products = self.catalogue_df.iloc[best_indices].reset_index(drop=True)
        return max_sims, matched_products

    def fit_transform(self, X_df):
        self.fit()
        return self.transform(X_df)