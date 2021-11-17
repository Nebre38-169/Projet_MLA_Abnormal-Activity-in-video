# Projet_MLA_Abnormal-Activity-in-video

#Comment utiliser Git et Github ?

Git est un gestionnaire de version et Github est un gestionnaire de dépot.
Les deux marches ensembles et permettent de suivre les évolutions d'un projet
et de travailler ensemble sans se gener.

Pour utiliser Github, juste rendez-vous sur le site de github.com et connectez vous.
Pour utiliser git, commencer par télécharger le logiciel à l'adresse : https://git-scm.com/download/win
Installer le sur votre machine.

Pour avoir acces au fichier sur votre machine, ouvrez l'explorateur de fichier et trouver le dossier dans lequel 
vous voulez stocker les fichiers. faites clic droit dans ce dossier et cliquez sur l'option "Git bash here".

Ensuite vous allez vous connecetez à votre compte github depuis git avec la commande "git config --global user.name [pseudo]" 
et "git config --global user.email [email]".
Pour récuperer le contenue du git, faites : "git clone [address]". L'adresse du git est trouvable sur le bouton vert au dessus des fichiers.

A ce moment la vous avez la dernière version du dépot sur votre machine. Si vous faite des modifications et que vous voulez les enregistrer, faites 
"git add ." pour ajouter les fichiers au prochain commit, puis "git commit". votre éditeur de texte s'ouvre et il faut alors décrire au mieux ce que 
vous ajoutez au dépot.

Pour envoyer les modifications au dépot distant (github), on va commencer par parametrer le dépot distant avec "git remote add [nom] [address]"
Mettez le nom que vous voulez mais l'adresse est la même que pour git clone.
Pour envoyer vos modifications, vous faites "git push [nom]" et cela envoye les derniers commit que vous avez fait qui ne sont pas encore sur le dépot distant.

Attention ! la dernière commande ne s'execute que si vous êtes vous même à jour sur les modifications.
Pour mettre à jour votre dépot local, vous faites "git pull [nom]".
