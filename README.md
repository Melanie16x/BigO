# Investigación de Algoritmos de notación Big O

1. **O(1) - Constante**

* Los algoritmos con complejidad O(1) realizan una operación cuya duración es independiente del tamaño de la entrada. Esto significa que no importa cuántos elementos tenga la entrada, el tiempo de ejecución permanece constante. Este tipo de algoritmos generalmente implica una operación simple, como acceder a un elemento en una estructura de datos que permite acceso directo, como un array.

2. **O(log n) - Logarítmica**

* Los algoritmos con complejidad O(log n) son aquellos en los que el tiempo de ejecución crece logarítmicamente con el tamaño de la entrada. Esto suele ocurrir en algoritmos que dividen repetidamente la entrada en partes más pequeñas, como en una búsqueda binaria. En cada paso, el tamaño del problema se reduce a la mitad, lo que conduce a un número de pasos que es proporcional al logaritmo del tamaño de la entrada.

3. **O(n) - Lineal**

* Los algoritmos con complejidad O(n) tienen un tiempo de ejecución que crece linealmente con el tamaño de la entrada. Esto significa que si se duplica el tamaño de la entrada, el tiempo de ejecución también se duplica. Los algoritmos lineales típicamente implican una operación que debe realizarse una vez por cada elemento de la entrada, como recorrer un array.

4. **O(n log n) - Linealítmica**

* Los algoritmos con complejidad O(n log n) combinan las características de complejidad lineal y logarítmica. Estos algoritmos suelen dividir la entrada en partes, resolver cada parte de manera recursiva y luego combinar los resultados. Los algoritmos de ordenamiento eficientes, como el ordenamiento por mezcla (merge sort), tienen esta complejidad. En estos algoritmos, la división y combinación contribuyen con el factor logarítmico, mientras que el procesamiento de cada división contribuye con el factor lineal.

5. **O(n^2) - Cuadrática**

* Los algoritmos con complejidad O(n^2) tienen un tiempo de ejecución que crece proporcionalmente al cuadrado del tamaño de la entrada. Esto es común en algoritmos que involucran bucles anidados, donde cada elemento de la entrada se compara o combina con cada otro elemento. Un ejemplo típico son los algoritmos de ordenamiento simples como el ordenamiento de burbuja.

6. **O(2^n) - Exponencial**

* Los algoritmos con complejidad O(2^n) tienen un tiempo de ejecución que crece exponencialmente con el tamaño de la entrada. Esto significa que el tiempo de ejecución se duplica con cada incremento en la entrada. Este tipo de complejidad es común en problemas que involucran todas las combinaciones posibles de la entrada, como ciertos problemas de toma de decisiones y de búsqueda en árboles, donde cada elección puede llevar a dos subproblemas diferentes.