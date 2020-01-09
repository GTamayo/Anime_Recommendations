# Anime Recommendations Database ¶
Recommendation data from 76,000 users at myanimelist.net 


El objetivo del desafío es desarrollar un sistema de recomendación que permita sugerir anime que los usuarios no han visto, en base a sus ratings previos.

Dataset:
https://www.kaggle.com/CooperUnion/anime-recommendations-database#rating.csv y descargar los archivos anime.csv y rating.csv. Ambas tablas contienen información de las preferencias de 73.516 usuarios en 12.294 animes (series de dibujos animados orientales). Cada usuario puede agregar un anime a su lista y darle un rating (de 0 a 10).

descripción de la data:

Anime.csv
• anime_id : id único del anime (de la página myanimelist.net)
• name : nombre del anime
• genre : lista de generos separados por coma del anime
• type : TV, movie (de película), OVA, etc...
• episodes : cantidad de episodios del show (1 si es película)
• rating : rating promedio (de 1-10) para este anime
• members : numero de miembros de la comunidad que están en el grupo del anime

Rating.csv
• user_id : id del usuario generado aleatoriamente
• anime_id : el anime que el usuario rankeo
• rating : el rating entre 1 y 10 que el usuario asignó al anime ( -1 si el usuario vio el anime pero no le asignó puntaje)

## Required

* Python Version 3.6.x.
* see requirement.txt

## Observacion

* Dependiendo de la configuración puede generarse un error de Allocate Memory producto de pivotar una tabla con una gran cantidad de registros, la solución más rapida para esto es trabajar con una muestra de registros en el pivoteo de la tabla. 
* Como mi local soporta realizar esta tarea el jupyter notebook commiteado esta ejecutado.
