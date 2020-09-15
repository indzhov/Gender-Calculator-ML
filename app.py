from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('gender.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    horror = request.form['a']
    thriller = request.form['b']
    comedy = request.form['c']
    romantic = request.form['d']
    sci_fi = request.form['e']
    war = request.form['k']
    fantasy = request.form['l']
    animation = request.form['m']
    documentary = request.form['g']
    western = request.form['w']
    action = request.form['n']
    flying = request.form['s1']
    storm = request.form['s2']
    darkness = request.form['s3']
    heights = request.form['s4']
    spiders = request.form['s5']
    snakes = request.form['s6']
    rats = request.form['s7']
    ageing = request.form['s8']
    dogs = request.form['s9']
    public_speaking = request.form['s10']
    history = request.form['i1']
    psychology = request.form['i2']
    politics = request.form['i3']
    mathematics = request.form['i4']
    physics = request.form['i5']
    internet = request.form['i6']
    pc = request.form['i7']
    economy_m = request.form['i8']
    biology = request.form['i9']
    chemistry = request.form['i10']
    reading = request.form['i11']
    geography = request.form['i12']
    leanguages = request.form['i13']
    medicine = request.form['i14']
    law = request.form['i15']
    cars = request.form['i16']
    art_exhibition = request.form['i17']
    religion = request.form['i18']
    countryside = request.form['i19']
    dancing = request.form['i20']
    m_instruments = request.form['i21']
    writing = request.form['i22']
    passive_sports = request.form['i23']
    active_sports = request.form['i24']
    gardening = request.form['i25']
    celebrities = request.form['i26']
    shopping = request.form['i27']
    s_and_t = request.form['i28']
    theatre = request.form['i29']
    friends = request.form['i30']
    adrenaline = request.form['i31']
    pets = request.form['i32']
    s_centres = request.form['m1']
    b_clothing = request.form['m2']
    entertainment = request.form['m3']
    looks = request.form['m4']
    gadgets = request.form['m5']
    healthy_e = request.form['m6']
    arr = np.array([[horror, thriller, comedy, romantic, sci_fi, war, fantasy, animation, documentary, western, action, flying, storm, darkness, heights, spiders, snakes, rats, ageing, dogs, public_speaking, history, psychology, politics, mathematics, physics, internet, pc, economy_m, biology, chemistry, reading, geography, leanguages, medicine, law, cars, art_exhibition, religion, countryside, dancing, m_instruments, writing, passive_sports, active_sports, gardening, celebrities, shopping, s_and_t, theatre, friends, adrenaline, pets, s_centres, b_clothing, entertainment, looks, gadgets, healthy_e]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















