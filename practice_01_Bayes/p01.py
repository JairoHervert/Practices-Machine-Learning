import pandas as pd

# 1. Cargamos los datos (el CSV que descargaste)
df = pd.read_csv('dataset_golf.csv')

def clasificador_paso_a_paso():
   # --- PASO 1: Calcular P(Vj) - Probabilidades a priori ---
   print("="*60)
   print("PASO 1: Calcular probabilidades a priori P(clase)")
   print("="*60)
   
   total = len(df)
   n_si = len(df[df['juego'] == 'sí'])
   n_no = len(df[df['juego'] == 'no'])
   
   p_si = n_si / total
   p_no = n_no / total
   
   print(f"Total de registros: {total}")
   print(f"Registros donde juego = 'sí': {n_si}")
   print(f"Registros donde juego = 'no': {n_no}")
   print(f"\nP(sí) = {n_si}/{total} = {p_si:.4f}")
   print(f"P(no) = {n_no}/{total} = {p_no:.4f}\n")

   # --- PASO 2: Entrada del usuario ---
   print("="*60)
   print("PASO 2: Ingresa los datos de la nueva entrada")
   print("="*60)
   clima_in = input("Clima (soleado/nublado/lluvioso): ").strip().lower()
   temp_in = input("Temperatura (calor/templado/frío): ").strip().lower()
   hum_in = input("Humedad (baja/normal/alta): ").strip().lower()
   viento_in = input("Viento (debil/fuerte): ").strip().lower()
   
   print(f"\nDatos ingresados:")
   print(f"  Clima: {clima_in}")
   print(f"  Temperatura: {temp_in}")
   print(f"  Humedad: {hum_in}")
   print(f"  Viento: {viento_in}\n")

   # --- PASO 3: Calcular P(ai | Vj) con Laplace Smoothing ---
   print("="*60)
   print("PASO 3: Calcular probabilidades condicionales P(característica|clase)")
   print("="*60)
   
   # Número de valores únicos para cada característica (para Laplace)
   n_clima = df['Clima'].nunique()
   n_temp = df['temperatura'].nunique()
   n_hum = df['humedad'].nunique()
   n_viento = df['viento'].nunique()
   
   # Filtramos por clase
   df_si = df[df['juego'] == 'sí']
   df_no = df[df['juego'] == 'no']
   
   # --- Para "SÍ" con Laplace Smoothing ---
   # Fórmula: (count + 1) / (total_clase + num_categorias)
   
   count_clima_si = len(df_si[df_si['Clima'] == clima_in])
   p_clima_si = (count_clima_si + 1) / (n_si + n_clima)
   
   count_temp_si = len(df_si[df_si['temperatura'] == temp_in])
   p_temp_si = (count_temp_si + 1) / (n_si + n_temp)
   
   count_hum_si = len(df_si[df_si['humedad'] == hum_in])
   p_hum_si = (count_hum_si + 1) / (n_si + n_hum)
   
   count_viento_si = len(df_si[df_si['viento'] == viento_in])
   p_viento_si = (count_viento_si + 1) / (n_si + n_viento)
   
   # --- Para "NO" con Laplace Smoothing ---
   count_clima_no = len(df_no[df_no['Clima'] == clima_in])
   p_clima_no = (count_clima_no + 1) / (n_no + n_clima)
   
   count_temp_no = len(df_no[df_no['temperatura'] == temp_in])
   p_temp_no = (count_temp_no + 1) / (n_no + n_temp)
   
   count_hum_no = len(df_no[df_no['humedad'] == hum_in])
   p_hum_no = (count_hum_no + 1) / (n_no + n_hum)
   
   count_viento_no = len(df_no[df_no['viento'] == viento_in])
   p_viento_no = (count_viento_no + 1) / (n_no + n_viento)

   # Mostramos los cálculos detallados
   print("\n--- Probabilidades condicionales para 'SÍ' ---")
   print(f"P(Clima={clima_in}|sí) = ({count_clima_si}+1) / ({n_si}+{n_clima}) = {p_clima_si:.4f}")
   print(f"P(Temp={temp_in}|sí) = ({count_temp_si}+1) / ({n_si}+{n_temp}) = {p_temp_si:.4f}")
   print(f"P(Hum={hum_in}|sí) = ({count_hum_si}+1) / ({n_si}+{n_hum}) = {p_hum_si:.4f}")
   print(f"P(Viento={viento_in}|sí) = ({count_viento_si}+1) / ({n_si}+{n_viento}) = {p_viento_si:.4f}")
   
   print("\n--- Probabilidades condicionales para 'NO' ---")
   print(f"P(Clima={clima_in}|no) = ({count_clima_no}+1) / ({n_no}+{n_clima}) = {p_clima_no:.4f}")
   print(f"P(Temp={temp_in}|no) = ({count_temp_no}+1) / ({n_no}+{n_temp}) = {p_temp_no:.4f}")
   print(f"P(Hum={hum_in}|no) = ({count_hum_no}+1) / ({n_no}+{n_hum}) = {p_hum_no:.4f}")
   print(f"P(Viento={viento_in}|no) = ({count_viento_no}+1) / ({n_no}+{n_viento}) = {p_viento_no:.4f}")

   # --- PASO 4: Calcular Vmap (Maximum A Posteriori) ---
   print("\n" + "="*60)
   print("PASO 4: Calcular V_MAP (multiplicar todas las probabilidades)")
   print("="*60)
   
   v_map_si = p_si * p_clima_si * p_temp_si * p_hum_si * p_viento_si
   v_map_no = p_no * p_clima_no * p_temp_no * p_hum_no * p_viento_no

   print("\n--- Cálculo para 'SÍ' ---")
   print(f"V_MAP(sí) = P(sí) × P(Clima|sí) × P(Temp|sí) × P(Hum|sí) × P(Viento|sí)")
   print(f"V_MAP(sí) = {p_si:.4f} × {p_clima_si:.4f} × {p_temp_si:.4f} × {p_hum_si:.4f} × {p_viento_si:.4f}")
   print(f"V_MAP(sí) = {v_map_si:.8f}")

   print("\n--- Cálculo para 'NO' ---")
   print(f"V_MAP(no) = P(no) × P(Clima|no) × P(Temp|no) × P(Hum|no) × P(Viento|no)")
   print(f"V_MAP(no) = {p_no:.4f} × {p_clima_no:.4f} × {p_temp_no:.4f} × {p_hum_no:.4f} × {p_viento_no:.4f}")
   print(f"V_MAP(no) = {v_map_no:.8f}")

   # --- PASO 5: Decisión ---
   print("\n" + "="*60)
   print("PASO 5: DECISIÓN FINAL")
   print("="*60)
   
   print(f"\nComparación:")
   print(f"  V_MAP(sí) = {v_map_si:.8f}")
   print(f"  V_MAP(no) = {v_map_no:.8f}")
   
   if v_map_si > v_map_no:
      porcentaje = (v_map_si / (v_map_si + v_map_no)) * 100
      print(f"\n✅ DECISIÓN: El valor máximo probable es 'SÍ'")
      print(f"   Confianza aproximada: {porcentaje:.2f}%")
   elif v_map_no > v_map_si:
      porcentaje = (v_map_no / (v_map_si + v_map_no)) * 100
      print(f"\n❌ DECISIÓN: El valor máximo probable es 'NO'")
      print(f"   Confianza aproximada: {porcentaje:.2f}%")
   else:
      print(f"\n⚖️  EMPATE: Ambas probabilidades son iguales")
   
   print("="*60 + "\n")

if __name__ == "__main__":
   clasificador_paso_a_paso()