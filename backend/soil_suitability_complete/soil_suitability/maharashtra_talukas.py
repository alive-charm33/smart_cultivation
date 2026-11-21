"""
Complete list of talukas (tehsils) in Maharashtra, grouped by district.
Source: Wikipedia "List of talukas of Maharashtra"
(https://en.wikipedia.org/wiki/List_of_talukas_of_Maharashtra), a
community-maintained administrative reference.

~358 talukas across 36 districts (the exact count varies slightly by
source depending on how city/urban talukas like Pimpri-Chinchwad are
counted). Unlike maharashtra_districts.py (36 points, hand-verified
coordinates), this file does NOT include coordinates -- taluka-level
coordinates aren't reliably available in one place, so
build_soil_cache.py geocodes each one automatically using OpenStreetMap's
free Nominatim service before fetching its soil data.

Each entry is (taluka_name, district_name) -- both needed for an
accurate geocoding lookup, since taluka names alone aren't unique
across India (e.g. "Karjat" appears in both Raigad and Ahmednagar).
"""

MAHARASHTRA_TALUKAS = [
    # Sindhudurg district
    ("Kankavli", "Sindhudurg"), ("Vaibhavwadi", "Sindhudurg"), ("Devgad", "Sindhudurg"),
    ("Malwan", "Sindhudurg"), ("Sawantwadi", "Sindhudurg"), ("Kudal", "Sindhudurg"),
    ("Vengurla", "Sindhudurg"), ("Dodamarg", "Sindhudurg"),
    # Ratnagiri district
    ("Ratnagiri", "Ratnagiri"), ("Sangameshwar", "Ratnagiri"), ("Lanja", "Ratnagiri"),
    ("Rajapur", "Ratnagiri"), ("Chiplun", "Ratnagiri"), ("Guhagar", "Ratnagiri"),
    ("Dapoli", "Ratnagiri"), ("Mandangad", "Ratnagiri"), ("Khed", "Ratnagiri"),
    # Raigad district
    ("Pen", "Raigad"), ("Alibag", "Raigad"), ("Murud", "Raigad"), ("Panvel", "Raigad"),
    ("Uran", "Raigad"), ("Karjat", "Raigad"), ("Khalapur", "Raigad"), ("Mangaon", "Raigad"),
    ("Tala", "Raigad"), ("Roha", "Raigad"), ("Sudhagad-Pali", "Raigad"), ("Mahad", "Raigad"),
    ("Poladpur", "Raigad"), ("Shrivardhan", "Raigad"), ("Mhasala", "Raigad"),
    # Mumbai City / Mumbai Suburban
    ("Mumbai City", "Mumbai City"),
    ("Bandra", "Mumbai Suburban"), ("Kurla", "Mumbai Suburban"),
    ("Andheri", "Mumbai Suburban"), ("Borivali", "Mumbai Suburban"),
    # Thane district
    ("Thane", "Thane"), ("Kalyan", "Thane"), ("Murbad", "Thane"), ("Bhiwandi", "Thane"),
    ("Shahapur", "Thane"), ("Ulhasnagar", "Thane"), ("Ambarnath", "Thane"),
    # Palghar district
    ("Palghar", "Palghar"), ("Vasai", "Palghar"), ("Dahanu", "Palghar"), ("Talasari", "Palghar"),
    ("Jawhar", "Palghar"), ("Mokhada", "Palghar"), ("Vada", "Palghar"), ("Vikramgad", "Palghar"),
    # Nashik district
    ("Nashik", "Nashik"), ("Igatpuri", "Nashik"), ("Dindori", "Nashik"), ("Peth", "Nashik"),
    ("Trimbakeshwar", "Nashik"), ("Kalwan", "Nashik"), ("Deola", "Nashik"), ("Surgana", "Nashik"),
    ("Baglan", "Nashik"), ("Malegaon", "Nashik"), ("Nandgaon", "Nashik"), ("Chandwad", "Nashik"),
    ("Niphad", "Nashik"), ("Sinnar", "Nashik"), ("Yeola", "Nashik"),
    # Nandurbar district
    ("Nandurbar", "Nandurbar"), ("Navapur", "Nandurbar"), ("Shahada", "Nandurbar"),
    ("Talode", "Nandurbar"), ("Akkalkuwa", "Nandurbar"), ("Dhadgaon", "Nandurbar"),
    # Dhule district
    ("Dhule", "Dhule"), ("Sakri", "Dhule"), ("Sindkheda", "Dhule"), ("Shirpur", "Dhule"),
    # Jalgaon district
    ("Jalgaon", "Jalgaon"), ("Jamner", "Jalgaon"), ("Erandol", "Jalgaon"), ("Dharangaon", "Jalgaon"),
    ("Bhusawal", "Jalgaon"), ("Raver", "Jalgaon"), ("Muktainagar", "Jalgaon"), ("Bodwad", "Jalgaon"),
    ("Yawal", "Jalgaon"), ("Amalner", "Jalgaon"), ("Parola", "Jalgaon"), ("Chopda", "Jalgaon"),
    ("Pachora", "Jalgaon"), ("Bhadgaon", "Jalgaon"), ("Chalisgaon", "Jalgaon"),
    # Buldhana district
    ("Buldhana", "Buldhana"), ("Chikhli", "Buldhana"), ("Deulgaon Raja", "Buldhana"),
    ("Jalgaon Jamod", "Buldhana"), ("Sangrampur", "Buldhana"), ("Malkapur", "Buldhana"),
    ("Motala", "Buldhana"), ("Nandura", "Buldhana"), ("Khamgaon", "Buldhana"),
    ("Shegaon", "Buldhana"), ("Mehkar", "Buldhana"), ("Sindkhed Raja", "Buldhana"),
    ("Lonar", "Buldhana"),
    # Akola district
    ("Akola", "Akola"), ("Akot", "Akola"), ("Telhara", "Akola"), ("Balapur", "Akola"),
    ("Patur", "Akola"), ("Murtajapur", "Akola"), ("Barshitakli", "Akola"),
    # Washim district
    ("Washim", "Washim"), ("Malegaon", "Washim"), ("Risod", "Washim"),
    ("Mangrulpir", "Washim"), ("Karanja", "Washim"), ("Manora", "Washim"),
    # Amravati district
    ("Amravati", "Amravati"), ("Bhatkuli", "Amravati"), ("Nandgaon Khandeshwar", "Amravati"),
    ("Dharni", "Amravati"), ("Chikhaldara", "Amravati"), ("Achalpur", "Amravati"),
    ("Chandurbazar", "Amravati"), ("Morshi", "Amravati"), ("Warud", "Amravati"),
    ("Daryapur", "Amravati"), ("Anjangaon-Surji", "Amravati"), ("Chandur", "Amravati"),
    ("Dhamangaon", "Amravati"), ("Tiosa", "Amravati"),
    # Wardha district
    ("Wardha", "Wardha"), ("Deoli", "Wardha"), ("Seloo", "Wardha"), ("Arvi", "Wardha"),
    ("Ashti", "Wardha"), ("Karanja", "Wardha"), ("Hinganghat", "Wardha"), ("Samudrapur", "Wardha"),
    # Nagpur district
    ("Nagpur Urban", "Nagpur"), ("Nagpur Rural", "Nagpur"), ("Kamptee", "Nagpur"),
    ("Hingna", "Nagpur"), ("Katol", "Nagpur"), ("Narkhed", "Nagpur"), ("Savner", "Nagpur"),
    ("Kalameshwar", "Nagpur"), ("Ramtek", "Nagpur"), ("Mouda", "Nagpur"), ("Parseoni", "Nagpur"),
    ("Umred", "Nagpur"), ("Kuhi", "Nagpur"), ("Bhiwapur", "Nagpur"),
    # Bhandara district
    ("Bhandara", "Bhandara"), ("Tumsar", "Bhandara"), ("Pauni", "Bhandara"),
    ("Mohadi", "Bhandara"), ("Sakoli", "Bhandara"), ("Lakhani", "Bhandara"), ("Lakhandur", "Bhandara"),
    # Gondia district
    ("Gondia", "Gondia"), ("Goregaon", "Gondia"), ("Salekasa", "Gondia"), ("Tiroda", "Gondia"),
    ("Amgaon", "Gondia"), ("Deori", "Gondia"), ("Arjuni-Morgaon", "Gondia"), ("Sadak-Arjuni", "Gondia"),
    # Gadchiroli district
    ("Gadchiroli", "Gadchiroli"), ("Dhanora", "Gadchiroli"), ("Chamorshi", "Gadchiroli"),
    ("Mulchera", "Gadchiroli"), ("Desaiganj", "Gadchiroli"), ("Armori", "Gadchiroli"),
    ("Kurkheda", "Gadchiroli"), ("Korchi", "Gadchiroli"), ("Aheri", "Gadchiroli"),
    ("Etapalli", "Gadchiroli"), ("Bhamragad", "Gadchiroli"), ("Sironcha", "Gadchiroli"),
    # Chandrapur district
    ("Chandrapur", "Chandrapur"), ("Saoli", "Chandrapur"), ("Mul", "Chandrapur"),
    ("Ballarpur", "Chandrapur"), ("Pombhurna", "Chandrapur"), ("Gondpimpri", "Chandrapur"),
    ("Warora", "Chandrapur"), ("Chimur", "Chandrapur"), ("Bhadravati", "Chandrapur"),
    ("Bramhapuri", "Chandrapur"), ("Nagbhid", "Chandrapur"), ("Sindewahi", "Chandrapur"),
    ("Rajura", "Chandrapur"), ("Korpana", "Chandrapur"), ("Jiwati", "Chandrapur"),
    # Yavatmal district
    ("Yavatmal", "Yavatmal"), ("Arni", "Yavatmal"), ("Babhulgaon", "Yavatmal"),
    ("Kalamb", "Yavatmal"), ("Darwha", "Yavatmal"), ("Digras", "Yavatmal"), ("Ner", "Yavatmal"),
    ("Pusad", "Yavatmal"), ("Umarkhed", "Yavatmal"), ("Mahagaon", "Yavatmal"),
    ("Kelapur", "Yavatmal"), ("Ralegaon", "Yavatmal"), ("Ghatanji", "Yavatmal"),
    ("Wani", "Yavatmal"), ("Maregaon", "Yavatmal"), ("Zari Jamani", "Yavatmal"),
    # Nanded district
    ("Nanded", "Nanded"), ("Ardhapur", "Nanded"), ("Mudkhed", "Nanded"), ("Bhokar", "Nanded"),
    ("Umri", "Nanded"), ("Loha", "Nanded"), ("Kandhar", "Nanded"), ("Kinwat", "Nanded"),
    ("Himayatnagar", "Nanded"), ("Hadgaon", "Nanded"), ("Mahur", "Nanded"), ("Deglur", "Nanded"),
    ("Mukhed", "Nanded"), ("Dharmabad", "Nanded"), ("Biloli", "Nanded"), ("Naigaon", "Nanded"),
    # Hingoli district
    ("Hingoli", "Hingoli"), ("Sengaon", "Hingoli"), ("Kalamnuri", "Hingoli"),
    ("Basmath", "Hingoli"), ("Aundha Nagnath", "Hingoli"),
    # Parbhani district
    ("Parbhani", "Parbhani"), ("Sonpeth", "Parbhani"), ("Gangakhed", "Parbhani"),
    ("Palam", "Parbhani"), ("Purna", "Parbhani"), ("Sailu", "Parbhani"), ("Jintur", "Parbhani"),
    ("Manwath", "Parbhani"), ("Pathri", "Parbhani"),
    # Jalna district
    ("Jalna", "Jalna"), ("Bhokardan", "Jalna"), ("Jafrabad", "Jalna"), ("Badnapur", "Jalna"),
    ("Ambad", "Jalna"), ("Ghansawangi", "Jalna"), ("Partur", "Jalna"), ("Mantha", "Jalna"),
    # Aurangabad (Chhatrapati Sambhajinagar) district
    ("Aurangabad", "Aurangabad"), ("Kannad", "Aurangabad"), ("Soegaon", "Aurangabad"),
    ("Sillod", "Aurangabad"), ("Phulambri", "Aurangabad"), ("Khuldabad", "Aurangabad"),
    ("Vaijapur", "Aurangabad"), ("Gangapur", "Aurangabad"), ("Paithan", "Aurangabad"),
    # Beed district
    ("Beed", "Beed"), ("Georai", "Beed"), ("Patoda", "Beed"), ("Shirur-Kasar", "Beed"),
    ("Ashti", "Beed"), ("Majalgaon", "Beed"), ("Wadwani", "Beed"), ("Kaij", "Beed"),
    ("Dharur", "Beed"), ("Parli", "Beed"), ("Ambajogai", "Beed"),
    # Latur district
    ("Latur", "Latur"), ("Renapur", "Latur"), ("Ausa", "Latur"), ("Ahmedpur", "Latur"),
    ("Jalkot", "Latur"), ("Chakur", "Latur"), ("Shirur Anantpal", "Latur"),
    ("Nilanga", "Latur"), ("Deoni", "Latur"), ("Udgir", "Latur"),
    # Osmanabad (Dharashiv) district
    ("Osmanabad", "Osmanabad"), ("Tuljapur", "Osmanabad"), ("Bhum", "Osmanabad"),
    ("Paranda", "Osmanabad"), ("Washi", "Osmanabad"), ("Kalamb", "Osmanabad"),
    ("Lohara", "Osmanabad"), ("Umarga", "Osmanabad"),
    # Solapur district
    ("Barshi", "Solapur"), ("Solapur North", "Solapur"), ("Solapur South", "Solapur"),
    ("Akkalkot", "Solapur"), ("Madha", "Solapur"), ("Karmala", "Solapur"),
    ("Pandharpur", "Solapur"), ("Mohol", "Solapur"), ("Malshiras", "Solapur"),
    ("Sangole", "Solapur"), ("Mangalvedhe", "Solapur"),
    # Ahmednagar district
    ("Ahmednagar", "Ahmednagar"), ("Shevgaon", "Ahmednagar"), ("Pathardi", "Ahmednagar"),
    ("Parner", "Ahmednagar"), ("Sangamner", "Ahmednagar"), ("Kopargaon", "Ahmednagar"),
    ("Akole", "Ahmednagar"), ("Shrirampur", "Ahmednagar"), ("Nevasa", "Ahmednagar"),
    ("Rahata", "Ahmednagar"), ("Rahuri", "Ahmednagar"), ("Shrigonda", "Ahmednagar"),
    ("Karjat", "Ahmednagar"), ("Jamkhed", "Ahmednagar"),
    # Pune district
    ("Pune City", "Pune"), ("Haveli", "Pune"), ("Khed", "Pune"), ("Junnar", "Pune"),
    ("Ambegaon", "Pune"), ("Maval", "Pune"), ("Mulshi", "Pune"), ("Shirur", "Pune"),
    ("Purandhar", "Pune"), ("Velhe", "Pune"), ("Bhor", "Pune"), ("Baramati", "Pune"),
    ("Indapur", "Pune"), ("Daund", "Pune"),
    # Satara district
    ("Satara", "Satara"), ("Jaoli", "Satara"), ("Koregaon", "Satara"), ("Wai", "Satara"),
    ("Mahabaleshwar", "Satara"), ("Khandala", "Satara"), ("Phaltan", "Satara"),
    ("Maan", "Satara"), ("Khatav", "Satara"), ("Patan", "Satara"), ("Karad", "Satara"),
    # Sangli district
    ("Miraj", "Sangli"), ("Kavathemahankal", "Sangli"), ("Tasgaon", "Sangli"),
    ("Jat", "Sangli"), ("Walwa", "Sangli"), ("Shirala", "Sangli"), ("Khanapur", "Sangli"),
    ("Atpadi", "Sangli"), ("Palus", "Sangli"), ("Kadegaon", "Sangli"),
    # Kolhapur district
    ("Karvir", "Kolhapur"), ("Panhala", "Kolhapur"), ("Shahuwadi", "Kolhapur"),
    ("Kagal", "Kolhapur"), ("Hatkanangale", "Kolhapur"), ("Shirol", "Kolhapur"),
    ("Radhanagari", "Kolhapur"), ("Gaganbawada", "Kolhapur"), ("Bhudargad", "Kolhapur"),
    ("Gadhinglaj", "Kolhapur"), ("Chandgad", "Kolhapur"), ("Ajra", "Kolhapur"),
]