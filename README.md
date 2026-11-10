# OP-ST Solution Explorer (view-only)

Interactive map of the 540-instance exact suite from the CS4234 project
"Budget Split Meets Geography". Every route shown was produced offline and
passed the shared validator; this app only displays saved results.

Run locally: `pip install -r requirements.txt` then
`streamlit run app/solution_explorer.py`.

Station geography: Oslo City Bike public GBFS data (NLOD 2.0).
Road travel times: OpenStreetMap contributors (ODbL).
Task states (rewards, service times) are synthetic.
