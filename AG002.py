import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

# Questão 2)
df = pd.read_csv('https://raw.githubusercontent.com/marcelovca90-inatel/AG002/main/palmerpenguins.csv')

# Questão 3)
df.replace({'Biscoe': 0, 'Dream': 1, 'Torgersen': 2,
            'FEMALE': 0, 'MALE': 1, 'Adelie': 0,
            'Chinstrap': 1, 'Gentoo': 2}, inplace=True)

# Questão 4)
new_df = df.reindex(columns=['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species'])
print(new_df)

# Questão 5)
X = new_df.drop(columns=['species'])
y = new_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Questão 7)
classifier = {
    "GNB": GaussianNB()
}
for name, model in classifier.items():
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # Questão 8)
    report = classification_report(y_test, y_pred_test)
    print(report)
    print(f"{name:15s} accuracy\ttrain = {train_accuracy:.1%} \ttest = {test_accuracy:.1%}")

# Questão 9)
user_data = []
user_data.append(input("Escolha a ilha (Biscoe, Dream, Torgersen): "))
user_data.append(input("Escolha o sexo (FEMALE, MALE): "))
user_data.append(float(input("Escolha culmen length (mm): ")))
user_data.append(float(input("Escolha culmen depth (mm): ")))
user_data.append(float(input("Escolha o comprimento da nadadeira (mm): ")))
user_data.append(float(input("Escolha a massa corporal (g): ")))

user_df = pd.DataFrame([user_data], columns=['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'])
user_df.replace({'Biscoe': 0, 'Dream': 1, 'Torgersen': 2, 'FEMALE': 0, 'MALE': 1}, inplace=True)

# Garantir que as colunas correspondam às usadas durante o treinamento
user_df = user_df[X_train.columns]

# Fazer a previsão
user_pred = model.predict(user_df)
species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
predicted_species = species_mapping[user_pred[0]]

print(f"A espécie prevista para os dados obtidos é: {predicted_species}")
