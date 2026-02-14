from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

internships = [
    # 🧠 Artificial Intelligence & Machine Learning
    {"role": "Machine Learning Intern", "company": "Google AI", "location": "Bengaluru, India",
     "skills": ["AI", "ML", "Deep Learning", "NLP", "Computer Vision"],
     "interests": ["Artificial Intelligence", "Neural Networks", "Deep Learning Applications"],
     "link": "https://ai.google/research/join-us"},
    {"role": "AI Research Intern", "company": "Microsoft Research", "location": "Hyderabad, India",
     "skills": ["AI", "ML", "Computer Vision", "Reinforcement Learning"],
     "interests": ["AI Research", "Robotics", "Computer Vision Projects"],
     "link": "https://www.microsoft.com/en-us/research/careers/"},
    {"role": "Deep Learning Intern", "company": "OpenAI", "location": "Remote",
     "skills": ["Deep Learning", "NLP", "Transformers"],
     "interests": ["Natural Language Processing", "AI Research", "Transformers"],
     "link": "https://openai.com/careers"},
    {"role": "Computer Vision Intern", "company": "Adobe Research", "location": "Noida, India",
     "skills": ["Computer Vision", "Image Processing", "ML"],
     "interests": ["Image Processing", "Computer Vision", "AI for Creative Tools"],
     "link": "https://research.adobe.com/careers/"},
    {"role": "NLP Intern", "company": "Amazon AI", "location": "Seattle, USA",
     "skills": ["NLP", "ML", "Text Analytics"],
     "interests": ["Natural Language Processing", "Text Mining", "AI Chatbots"],
     "link": "https://www.amazon.jobs/en/teams/AI"},
    {"role": "AI & Robotics Intern", "company": "Tesla AI", "location": "Palo Alto, USA",
     "skills": ["AI", "Robotics", "Computer Vision"],
     "interests": ["Autonomous Vehicles", "Robotics", "AI Systems"],
     "link": "https://www.tesla.com/careers/search#/"},

    # 📊 Data Science & Analytics
    {"role": "Data Science Intern", "company": "IBM", "location": "Bengaluru, India",
     "skills": ["Big Data", "Statistics", "Predictive Modeling", "Python", "SQL"],
     "interests": ["Data Analysis", "Predictive Analytics", "Machine Learning"],
     "link": "https://www.ibm.com/in-en/employment/"},
    {"role": "Data Analyst Intern", "company": "Accenture", "location": "Mumbai, India",
     "skills": ["Data Analysis", "Python", "SQL", "Visualization"],
     "interests": ["Data Visualization", "Business Intelligence", "Analytics Projects"],
     "link": "https://www.accenture.com/in-en/careers"},
    {"role": "Business Analytics Intern", "company": "Deloitte", "location": "Bengaluru, India",
     "skills": ["Analytics", "Statistics", "Excel", "SQL"],
     "interests": ["Business Analytics", "Data-driven Strategy", "Reporting"],
     "link": "https://www2.deloitte.com/in/en/careers.html"},
    {"role": "Data Science Intern", "company": "Cognizant", "location": "Chennai, India",
     "skills": ["ML", "Big Data", "Python", "R"],
     "interests": ["Machine Learning", "Big Data Analysis", "Predictive Models"],
     "link": "https://careers.cognizant.com/global/en"},
    {"role": "Data Analytics Intern", "company": "Flipkart", "location": "Bengaluru, India",
     "skills": ["Data Analytics", "Python", "SQL", "Visualization"],
     "interests": ["E-commerce Analytics", "Data-driven Decision Making", "Visualization"],
     "link": "https://www.flipkartcareers.com/"},
    {"role": "Machine Learning & Data Intern", "company": "Fractal Analytics", "location": "Mumbai, India",
     "skills": ["ML", "Statistics", "Python", "Predictive Modeling"],
     "interests": ["AI in Business", "Predictive Analytics", "Data Science Projects"],
     "link": "https://fractal.ai/careers/"},

    # 🌐 Web & App Development
    {"role": "Frontend Developer Intern", "company": "Infosys", "location": "Pune, India",
     "skills": ["HTML", "CSS", "JavaScript", "React", "Frontend Development"],
     "interests": ["Web Development", "UI/UX Design", "Frontend Frameworks"],
     "link": "https://www.infosys.com/careers/"},
    {"role": "Full Stack Developer Intern", "company": "TCS", "location": "Mumbai, India",
     "skills": ["Full Stack", "Node.js", "React", "MongoDB"],
     "interests": ["Full Stack Development", "Web Applications", "Backend Systems"],
     "link": "https://www.tcs.com/careers"},
    {"role": "Mobile App Developer Intern", "company": "Swiggy", "location": "Bengaluru, India",
     "skills": ["Android", "iOS", "Flutter", "React Native"],
     "interests": ["Mobile App Development", "UI/UX Design", "E-commerce Apps"],
     "link": "https://careers.swiggy.com/"},
    {"role": "Web Developer Intern", "company": "Zomato", "location": "Gurgaon, India",
     "skills": ["HTML", "CSS", "JavaScript", "Backend"],
     "interests": ["Web Applications", "Frontend/Backend Development", "User Experience"],
     "link": "https://www.zomato.com/careers"},
    {"role": "Full Stack Intern", "company": "Paytm", "location": "Noida, India",
     "skills": ["Full Stack", "Node.js", "React", "Database"],
     "interests": ["Full Stack Web Development", "Payment Systems", "App Development"],
     "link": "https://paytm.com/careers/"},
    {"role": "Frontend Intern", "company": "Ola", "location": "Bengaluru, India",
     "skills": ["React", "CSS", "HTML", "JavaScript"],
     "interests": ["Web UI Development", "Frontend Technologies", "App Interfaces"],
     "link": "https://www.olacabs.com/careers"},

    # ☁️ Cloud & DevOps
    {"role": "Cloud Engineer Intern", "company": "AWS", "location": "Seattle, USA",
     "skills": ["AWS", "DevOps", "Kubernetes", "Docker"],
     "interests": ["Cloud Computing", "Infrastructure Automation", "Scalable Systems"],
     "link": "https://aws.amazon.com/careers/"},
    {"role": "DevOps Intern", "company": "Microsoft Azure", "location": "Hyderabad, India",
     "skills": ["Azure", "CI/CD", "Docker", "Kubernetes"],
     "interests": ["Cloud Deployment", "Automation", "DevOps Practices"],
     "link": "https://careers.microsoft.com/"},
    {"role": "Cloud Solutions Intern", "company": "Google Cloud", "location": "Bengaluru, India",
     "skills": ["GCP", "Cloud Architecture", "DevOps Tools"],
     "interests": ["Cloud Platforms", "Automation", "Enterprise Solutions"],
     "link": "https://careers.google.com/"},
    {"role": "Kubernetes Intern", "company": "Red Hat", "location": "Pune, India",
     "skills": ["Kubernetes", "Docker", "Cloud Services"],
     "interests": ["Containerization", "Cloud Orchestration", "DevOps"],
     "link": "https://www.redhat.com/en/jobs"},
    {"role": "Cloud & DevOps Intern", "company": "IBM Cloud", "location": "Bengaluru, India",
     "skills": ["Cloud", "Docker", "CI/CD", "DevOps"],
     "interests": ["Cloud Automation", "Infrastructure Management", "Deployment Pipelines"],
     "link": "https://www.ibm.com/in-en/employment/"},
    {"role": "Cloud Operations Intern", "company": "Oracle Cloud", "location": "Bengaluru, India",
     "skills": ["Cloud", "Monitoring", "CI/CD", "Kubernetes"],
     "interests": ["Cloud Operations", "Automation", "Enterprise Systems"],
     "link": "https://www.oracle.com/corporate/careers/"},

    # 🔒 Cybersecurity & Ethical Hacking
    {"role": "Cybersecurity Intern", "company": "Palo Alto Networks", "location": "USA",
     "skills": ["Network Security", "PenTesting", "Firewalls"],
     "interests": ["Cybersecurity", "Threat Analysis", "Penetration Testing"],
     "link": "https://www.paloaltonetworks.com/company/careers"},
    {"role": "Ethical Hacking Intern", "company": "Kaspersky", "location": "Moscow, Russia",
     "skills": ["Ethical Hacking", "PenTesting", "Network Security"],
     "interests": ["Cybersecurity Research", "Vulnerability Analysis", "Ethical Hacking"],
     "link": "https://www.kaspersky.com/about/careers"},
    {"role": "Cybersecurity Analyst Intern", "company": "Symantec", "location": "Bengaluru, India",
     "skills": ["Cybersecurity", "Network Analysis", "Security Protocols"],
     "interests": ["Threat Mitigation", "Network Security", "Cyber Defense"],
     "link": "https://www.broadcom.com/company/careers"},
    {"role": "Information Security Intern", "company": "Infosys", "location": "Pune, India",
     "skills": ["Security Policies", "PenTesting", "Vulnerability Assessment"],
     "interests": ["Information Security", "PenTesting", "Cyber Defense Strategies"],
     "link": "https://www.infosys.com/careers/"},
    {"role": "Cybersecurity Intern", "company": "TCS", "location": "Mumbai, India",
     "skills": ["Cybersecurity", "Risk Assessment", "Network Security"],
     "interests": ["Cybersecurity Operations", "Threat Detection", "Security Research"],
     "link": "https://www.tcs.com/careers"},
    {"role": "Security Operations Intern", "company": "Capgemini", "location": "Bengaluru, India",
     "skills": ["Security Monitoring", "Incident Response", "PenTesting"],
     "interests": ["Security Operations", "Incident Handling", "Network Security"],
     "link": "https://www.capgemini.com/careers/"},

    # 🧱 Blockchain & Web3
    {"role": "Blockchain Developer Intern", "company": "Consensys", "location": "Remote",
     "skills": ["Solidity", "Smart Contracts", "Ethereum"],
     "interests": ["Blockchain Development", "Decentralized Apps", "Smart Contracts"],
     "link": "https://consensys.net/careers/"},
    {"role": "Web3 Developer Intern", "company": "Polygon", "location": "Bengaluru, India",
     "skills": ["Blockchain", "DApps", "Solidity"],
     "interests": ["Web3 Applications", "Smart Contracts", "DeFi Projects"],
     "link": "https://polygon.technology/careers/"},
    {"role": "Blockchain Intern", "company": "IBM Blockchain", "location": "Bengaluru, India",
     "skills": ["Hyperledger", "Blockchain", "Smart Contracts"],
     "interests": ["Enterprise Blockchain", "Supply Chain Blockchain", "Smart Contracts"],
     "link": "https://www.ibm.com/in-en/employment/"},
    {"role": "DeFi Intern", "company": "Uniswap Labs", "location": "Remote",
     "skills": ["DeFi", "Smart Contracts", "Ethereum"],
     "interests": ["Decentralized Finance", "Blockchain Applications", "Smart Contract Development"],
     "link": "https://uniswap.org/careers/"},
    {"role": "Blockchain Research Intern", "company": "Coinbase", "location": "San Francisco, USA",
     "skills": ["Blockchain", "Crypto", "DApps"],
     "interests": ["Cryptocurrency", "Blockchain Research", "Decentralized Systems"],
     "link": "https://www.coinbase.com/careers"},
    {"role": "Blockchain Engineering Intern", "company": "Chainlink Labs", "location": "Remote",
     "skills": ["Blockchain", "Oracles", "Smart Contracts"],
     "interests": ["Decentralized Oracles", "Blockchain Engineering", "Smart Contracts"],
     "link": "https://chain.link/careers"},

    # ⚛️ Quantum Computing
    {"role": "Quantum Computing Intern", "company": "IBM Q", "location": "Yorktown Heights, USA",
     "skills": ["Qiskit", "Quantum Algorithms", "Python"],
     "interests": ["Quantum Computing", "Quantum Algorithms", "Qiskit Development"],
     "link": "https://www.ibm.com/quantum-computing/careers/"},
    {"role": "Quantum Research Intern", "company": "Microsoft Quantum", "location": "Redmond, USA",
     "skills": ["Quantum Computing", "Quantum Algorithms", "Python"],
     "interests": ["Quantum Research", "Quantum Simulation", "Q# Programming"],
     "link": "https://careers.microsoft.com/"},
    {"role": "Quantum Software Intern", "company": "Google Quantum AI", "location": "Santa Barbara, USA",
     "skills": ["Quantum Algorithms", "Python", "Quantum Circuits"],
     "interests": ["Quantum Software", "Algorithm Design", "Quantum Computing Research"],
     "link": "https://careers.google.com/"},
    {"role": "Quantum Developer Intern", "company": "D-Wave Systems", "location": "Burnaby, Canada",
     "skills": ["Quantum Computing", "Python", "Optimization Algorithms"],
     "interests": ["Quantum Annealing", "Optimization Problems", "Quantum Research"],
     "link": "https://www.dwavesys.com/careers"},
    {"role": "Quantum Algorithms Intern", "company": "Rigetti Computing", "location": "Berkeley, USA",
     "skills": ["Quantum Algorithms", "Qiskit", "Python"],
     "interests": ["Quantum Algorithm Development", "Quantum Simulation", "Quantum Systems"],
     "link": "https://www.rigetti.com/careers"},
    {"role": "Quantum Computing Intern", "company": "Xanadu", "location": "Toronto, Canada",
     "skills": ["Quantum Software", "QML", "Python"],
     "interests": ["Quantum Machine Learning", "Quantum Software", "Research Projects"],
     "link": "https://www.xanadu.ai/careers"},

    # 🕹️ AR/VR & Game Development
    {"role": "Game Development Intern", "company": "Ubisoft", "location": "Montreuil, France",
     "skills": ["Unity", "Unreal Engine", "C#"],
     "interests": ["Game Design", "AR/VR Development", "3D Graphics"],
     "link": "https://www.ubisoft.com/careers"},
    {"role": "VR/AR Developer Intern", "company": "Oculus VR", "location": "Menlo Park, USA",
     "skills": ["Unity", "VR", "C++"],
     "interests": ["Virtual Reality", "AR/VR Apps", "Immersive Experiences"],
     "link": "https://www.oculus.com/careers/"},
    {"role": "XR Developer Intern", "company": "Niantic", "location": "San Francisco, USA",
     "skills": ["Unity", "AR", "C#"],
     "interests": ["Augmented Reality", "Game Development", "AR Apps"],
     "link": "https://nianticlabs.com/careers/"},
    {"role": "Unity Developer Intern", "company": "Electronic Arts", "location": "California, USA",
     "skills": ["Unity", "C#", "Game Programming"],
     "interests": ["Game Programming", "3D Simulation", "VR/AR Games"],
     "link": "https://www.ea.com/careers"},
    {"role": "Game Design Intern", "company": "Activision", "location": "Santa Monica, USA",
     "skills": ["C++", "Game Mechanics", "Unity"],
     "interests": ["Game Design", "Game Mechanics", "Interactive Media"],
     "link": "https://careers.activision.com/"},
    {"role": "VR Intern", "company": "Google AR/VR", "location": "Mountain View, USA",
     "skills": ["Unity", "VR", "ARCore"],
     "interests": ["Augmented & Virtual Reality", "XR Applications", "3D Development"],
     "link": "https://careers.google.com/"},
]

# ⚙️ Embedded Systems & IoT
internships += [
    {"role": "Embedded Systems Intern", "company": "Texas Instruments", "location": "Bengaluru, India",
     "skills": ["Microcontrollers", "Embedded C", "IoT"],
     "interests": ["IoT Devices", "Microcontroller Programming", "Edge Computing"],
     "link": "https://careers.ti.com/"},
    {"role": "IoT Developer Intern", "company": "Bosch", "location": "Bengaluru, India",
     "skills": ["IoT", "Raspberry Pi", "Sensors", "Python"],
     "interests": ["IoT Solutions", "Embedded Systems", "Hardware-Software Integration"],
     "link": "https://www.bosch-career.com/"},
    {"role": "Embedded Firmware Intern", "company": "STMicroelectronics", "location": "Noida, India",
     "skills": ["Firmware Development", "C/C++", "Microcontrollers"],
     "interests": ["Embedded Firmware", "Hardware Programming", "IoT Devices"],
     "link": "https://www.st.com/content/st_com/en/about/careers.html"},
    {"role": "IoT Systems Intern", "company": "Honeywell", "location": "Pune, India",
     "skills": ["IoT", "Edge Computing", "Sensors"],
     "interests": ["Edge Computing", "IoT Networks", "Hardware Integration"],
     "link": "https://careers.honeywell.com/"},
    {"role": "Embedded Hardware Intern", "company": "Intel", "location": "Bengaluru, India",
     "skills": ["Microcontrollers", "Embedded Systems", "C/C++"],
     "interests": ["Embedded Hardware", "IoT Applications", "Edge Computing"],
     "link": "https://www.intel.com/content/www/us/en/jobs/jobs-at-intel.html"},
    {"role": "IoT Intern", "company": "Siemens", "location": "Mumbai, India",
     "skills": ["IoT", "Microcontrollers", "Python", "Edge Computing"],
     "interests": ["Industrial IoT", "Embedded Systems", "Automation"],
     "link": "https://new.siemens.com/global/en/company/jobs.html"},
]

# 🤖 Robotics & Automation
internships += [
    {"role": "Robotics Intern", "company": "ABB Robotics", "location": "Pune, India",
     "skills": ["ROS", "Control Systems", "Robotics"],
     "interests": ["Industrial Robotics", "Automation", "Robot Programming"],
     "link": "https://new.abb.com/careers"},
    {"role": "Automation Engineer Intern", "company": "Siemens", "location": "Bengaluru, India",
     "skills": ["PLC", "Control Systems", "Automation"],
     "interests": ["Automation Systems", "Robotics", "Industrial Control"],
     "link": "https://new.siemens.com/global/en/company/jobs.html"},
    {"role": "ROS Developer Intern", "company": "Fetch Robotics", "location": "San Jose, USA",
     "skills": ["ROS", "Robot Programming", "Python", "C++"],
     "interests": ["Robotics Software", "ROS Development", "Autonomous Systems"],
     "link": "https://fetchrobotics.com/careers/"},
    {"role": "Drone Systems Intern", "company": "DJI", "location": "Shenzhen, China",
     "skills": ["Drone Programming", "ROS", "C++"],
     "interests": ["Autonomous Drones", "Robotics", "Control Systems"],
     "link": "https://www.dji.com/careers"},
    {"role": "Robotics Engineer Intern", "company": "Boston Dynamics", "location": "Waltham, USA",
     "skills": ["Robotics", "Control Systems", "Python", "C++"],
     "interests": ["Robotics Engineering", "Automation", "Mechanical Systems"],
     "link": "https://www.bostondynamics.com/careers"},
    {"role": "Automation Intern", "company": "Rockwell Automation", "location": "Bengaluru, India",
     "skills": ["Automation", "PLC", "Control Systems"],
     "interests": ["Industrial Automation", "Robotics Integration", "Control Engineering"],
     "link": "https://www.rockwellautomation.com/en-us/company/careers.html"},
]

# 🧩 Software Engineering & Systems
internships += [
    {"role": "Software Engineering Intern", "company": "Google", "location": "Bengaluru, India",
     "skills": ["C++", "Distributed Systems", "Operating Systems"],
     "interests": ["System Programming", "Distributed Computing", "Software Engineering"],
     "link": "https://careers.google.com/"},
    {"role": "Backend Developer Intern", "company": "Microsoft", "location": "Hyderabad, India",
     "skills": ["C#", "Distributed Systems", "Cloud Systems"],
     "interests": ["Software Development", "Backend Systems", "Cloud Architecture"],
     "link": "https://careers.microsoft.com/"},
    {"role": "Systems Intern", "company": "Red Hat", "location": "Pune, India",
     "skills": ["Linux", "C/C++", "Distributed Systems"],
     "interests": ["Open Source Systems", "Linux Kernel", "Distributed Systems"],
     "link": "https://www.redhat.com/en/jobs"},
    {"role": "Software Developer Intern", "company": "Amazon", "location": "Bengaluru, India",
     "skills": ["Java", "C++", "Distributed Systems"],
     "interests": ["Backend Development", "High Performance Systems", "Distributed Computing"],
     "link": "https://www.amazon.jobs/"},
    {"role": "Systems Software Intern", "company": "Intel", "location": "Bengaluru, India",
     "skills": ["C/C++", "Operating Systems", "Low-level Programming"],
     "interests": ["System Programming", "Operating Systems", "Performance Optimization"],
     "link": "https://www.intel.com/content/www/us/en/jobs/jobs-at-intel.html"},
    {"role": "Full Stack Systems Intern", "company": "Facebook/Meta", "location": "Hyderabad, India",
     "skills": ["C++", "Distributed Systems", "Web Development"],
     "interests": ["Full Stack Systems", "Backend & Frontend", "Distributed Architecture"],
     "link": "https://www.metacareers.com/"},
]

# 💡 Research, Open Source & Fellowships
internships += [
    {"role": "GSoC Intern", "company": "Google Summer of Code", "location": "Remote",
     "skills": ["Open Source Development", "Python", "C++"],
     "interests": ["Open Source Contribution", "Software Development", "Community Projects"],
     "link": "https://summerofcode.withgoogle.com/"},
    {"role": "CERN Research Intern", "company": "CERN", "location": "Geneva, Switzerland",
     "skills": ["Physics Simulation", "Data Analysis", "Python", "C++"],
     "interests": ["Particle Physics", "Research Projects", "High Energy Physics"],
     "link": "https://home.cern/careers/internships"},
    {"role": "NASA Research Intern", "company": "NASA", "location": "USA",
     "skills": ["Research", "Python", "Data Analysis", "Simulation"],
     "interests": ["Space Research", "Aerospace Projects", "Scientific Computing"],
     "link": "https://intern.nasa.gov/"},
    {"role": "Open Source Intern", "company": "Linux Foundation", "location": "Remote",
     "skills": ["Open Source", "C/C++", "Python"],
     "interests": ["Open Source Development", "Linux Kernel", "Community Projects"],
     "link": "https://www.linuxfoundation.org/careers/"},
    {"role": "Research Intern", "company": "MIT CSAIL", "location": "Cambridge, USA",
     "skills": ["AI", "Robotics", "Python", "C++"],
     "interests": ["AI Research", "Robotics Research", "Computer Science Projects"],
     "link": "https://www.csail.mit.edu/"},
]

# 🧬 Bioinformatics & Computational Biology
internships += [
    {"role": "Bioinformatics Intern", "company": "NCBI", "location": "Bethesda, USA",
     "skills": ["Genomics", "Python", "R", "Data Analysis"],
     "interests": ["Computational Biology", "Genomics", "Healthcare AI"],
     "link": "https://www.ncbi.nlm.nih.gov/about/careers/"},
    {"role": "Computational Biology Intern", "company": "Broad Institute", "location": "Cambridge, USA",
     "skills": ["Bioinformatics", "Python", "Data Analysis", "Machine Learning"],
     "interests": ["Genomics", "Bioinformatics Research", "AI in Healthcare"],
     "link": "https://www.broadinstitute.org/careers"},
    {"role": "AI in Healthcare Intern", "company": "IBM Watson Health", "location": "Bengaluru, India",
     "skills": ["AI", "Python", "Healthcare Data Analysis"],
     "interests": ["AI in Medicine", "Healthcare Analytics", "Predictive Modeling"],
     "link": "https://www.ibm.com/watson-health/careers/"},
    {"role": "Genomics Research Intern", "company": "Illumina", "location": "San Diego, USA",
     "skills": ["Genomics", "Data Analysis", "Python", "R"],
     "interests": ["Genome Sequencing", "Bioinformatics", "Healthcare Innovation"],
     "link": "https://www.illumina.com/company/careers.html"},
    {"role": "Computational Biology Intern", "company": "Emory University", "location": "Atlanta, USA",
     "skills": ["Bioinformatics", "Data Analysis", "Python", "R"],
     "interests": ["Bioinformatics Research", "Genomics", "Healthcare AI"],
     "link": "https://www.emory.edu/home/careers.html"},
]


model = None
chat_session = None

def preprocess(text):
    if not text:
        return ""
    df = pd.DataFrame({"input": [text]})
    df["input"] = (
        df["input"]
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df["input"].iloc[0]

def recommend_internships(user_input):
    user_input = preprocess(user_input)
    corpus = [" ".join(intern["skills"] + intern["interests"]) for intern in internships]
    corpus.append(user_input)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    results = []
    for idx, score in enumerate(similarity[0]):
        results.append({**internships[idx], "match": round(score * 100, 2)})
    return sorted(results, key=lambda x: x["match"], reverse=True)

@app.route("/")
def serve_frontend():
    return send_from_directory(".", "index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.json.get("skills", "")
    recommendations = recommend_internships(user_input)
    return jsonify(recommendations)

@app.route("/chat", methods=["POST"])
def chat():
    global chat_session
    user_message = request.json.get("message", "")
    if chat_session is None:
        return jsonify({"reply": "AI is not initialized. Please check the server logs."})
    try:
        response = chat_session.send_message(user_message)
        return jsonify({"reply": response.text})
    except Exception:
        return jsonify({"reply": "Sorry, something went wrong with the AI response."})

if __name__ == "__main__":
    try:
        api_key = "AIzaSyBWUBuNOdXL0nOu8u927FhX0wEbq8rEi58"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat_session = model.start_chat(history=[])
        print("AI model successfully initialized.")
    except Exception as e:
        print(f"âŒ Initialization error: {e}")
        exit()
    app.run(debug=True,port=8674)