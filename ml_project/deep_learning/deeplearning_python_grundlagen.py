# Python Grundlagen

class Auto: 
    def __init__(self, leistung, hubraum, marke, atkuelle_geschwindigkeit=0, benzinstand=100):
        self.leistung = leistung
        self.hubraum = hubraum
        self.marke = marke
        self.atkuelle_geschwindigkeit = atkuelle_geschwindigkeit
        self.benzinstand = benzinstand

    def beschleunigen(self, menge):
        self.atkuelle_geschwindigkeit += menge
        
    def bremsen(self, menge):
        self.atkuelle_geschwindigkeit -= menge
        if self.atkuelle_geschwindigkeit < 0:
            self.atkuelle_geschwindigkeit = 0  

    def tanken(self, menge):
        self.benzinstand += menge
        if self.benzinstand > 100:
            self.benzinstand = 100
    
mein_auto = Auto(150, 2000, "BMW")