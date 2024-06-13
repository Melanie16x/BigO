import time
import matplotlib.pyplot as plt
import numpy as np

# Algoritmos de ejemplo

# O(1) - Constante
def constant_algorithm(n):
    arr = np.random.randint(1, 1000, n)
    return arr[0]

# O(log n) - Logarítmica
def binary_search(n):
    arr = sorted(np.random.randint(1, 1000, n))
    x = arr[n//2]
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            l = mid + 1
        else:
            r = mid - 1
    return -1

# O(n) - Lineal
def linear_search(n):
    arr = np.random.randint(1, 1000, n)
    x = arr[n//2]
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1

# O(n log n) - Linealítmica
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

def merge_sort_wrapper(n):
    arr = np.random.randint(1, 1000, n)
    merge_sort(arr)

# O(n^2) - Cuadrática
def bubble_sort(n):
    arr = np.random.randint(1, 1000, n)
    for i in range(len(arr)):
        for j in range(0, len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

# O(2^n) - Exponencial
def fibonacci_recursive(n):
    if n <= 1:
        return n
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# Función para medir tiempos de ejecución
def measure_time(func, n):
    start_time = time.time()
    func(n)
    end_time = time.time()
    return end_time - start_time

# Graficar la complejidad de los algoritmos
sizes = [10, 20, 40, 80, 160, 320, 640]
sizes_exponential = [5, 10, 15, 20, 25, 30]

times_constant = [measure_time(constant_algorithm, n) for n in sizes]
times_logarithmic = [measure_time(binary_search, n) for n in sizes]
times_linear = [measure_time(linear_search, n) for n in sizes]
times_linearithmic = [measure_time(merge_sort_wrapper, n) for n in sizes]
times_quadratic = [measure_time(bubble_sort, n) for n in sizes]
times_exponential = [measure_time(fibonacci_recursive, n) for n in sizes_exponential]

# Graficar los resultados
plt.figure(figsize=(12, 8))
plt.plot(sizes, times_constant, label="O(1) - Constante")
plt.plot(sizes, times_logarithmic, label="O(log n) - Logarítmica")
plt.plot(sizes, times_linear, label="O(n) - Lineal")
plt.plot(sizes, times_linearithmic, label="O(n log n) - Linealítmica")
plt.plot(sizes, times_quadratic, label="O(n^2) - Cuadrática")
plt.plot(sizes_exponential, times_exponential, label="O(2^n) - Exponencial")

plt.xlabel('Tamaño de la entrada (n)')
plt.ylabel('Tiempo de ejecución (segundos)')
plt.title('Comparación de la complejidad temporal de diferentes algoritmos')
plt.legend()
plt.yscale('log')  # Usamos escala logarítmica para visualizar mejor las diferencias
plt.grid(True)
plt.show()
