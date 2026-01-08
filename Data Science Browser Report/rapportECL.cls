\LoadClass[a4paper,10pt]{article} % Classe du document

%--------------------- Packages ------------------------

\RequirePackage{multicol}
\RequirePackage[english]{babel} %Langue du document
\RequirePackage[utf8]{inputenc} %Caractères spéciaux
\RequirePackage[section]{placeins}%Pour placement de section
\RequirePackage[T1]{fontenc} %Quelques lettres qui sont pas inclus dans UTF-8
\RequirePackage{mathtools} %Paquet pour des équations et symboles mathématiques
\RequirePackage{siunitx} %Pour écrire avec la notation scientifique (Ex.: \num{2e+9})
\RequirePackage{float} %Pour placement d'images
\RequirePackage{graphicx} %Paquet pour insérer des images
\RequirePackage[justification=centering]{caption} %Pour les légendes centralisées
\RequirePackage{subcaption}
\RequirePackage{wallpaper}
\RequirePackage{nomencl}
%\makenomenclature
\RequirePackage{fancyhdr}
%\pagestyle{fancy}
%\fancyheadoffset{1cm}
%\setlength{\headheight}{2cm}
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref}%Paquet pour insérer légendes dans des sous-figures comme Figure 1a, 1b
\RequirePackage[left=2.5cm,right=2.5cm,top=1cm,bottom=3.5cm]{geometry} %Configuration de la page


%-------------------- Informations sur le rapport ----------------------

\newcommand{\UE}[1]{\renewcommand{\UE}{#1}}
\newcommand{\sujet}[1]{\renewcommand{\sujet}{#1}}
\newcommand{\titre}[1]{\renewcommand{\titre}{#1}}
\newcommand{\enseignant}[1]{\renewcommand{\enseignant}{#1}}
\newcommand{\eleves}[1]{\renewcommand{\eleves}{#1}}

\newcommand{\fairemarges}{
\makenomenclature
\pagestyle{fancy}
\fancyheadoffset{0cm}
\setlength{\headheight}{2cm}
\lhead{\includegraphics[scale=0.3]{logos/logo_ECL.jpg}} %Affichage de l'image au top de la page
\rhead{\nouppercase{\leftmark}}
\rfoot{\thepage}
\cfoot{Rapport - \textbf{\titre}}
\lfoot{\UE}
}

\newcommand{\fairepagedegarde}{
\begin{titlepage}

\ThisLRCornerWallPaper{0.6}{logos/SteinECL.jpg}
	\centering %Centraliser le contenu
	\includegraphics[width=0.4\textwidth]{logos/logo_ECL.jpg}\par\vspace{1cm} %Insertion du logo
	{\scshape\LARGE École Centrale de Lyon\par} %Nom de l'université
	\vspace{1.5cm}%Espace de 1,5cm
	{\scshape\Large \UE \\ \sujet \\ Report\par} %sous-titre
	\vspace{1cm}%Espace de 1cm
    \rule{\linewidth}{0.2 mm} \\[0.4 cm]
	{\huge\bfseries \titre \par} \
    \rule{\linewidth}{0.2 mm} \\[1.5 cm]
	\vspace{1cm}%Espace de 3cm
    
	\begin{minipage}{0.5\textwidth} %Minipage pour faire deux colonnes
		\begin{flushleft} \large %Envoyer à gauche
		\emph{\textbf{Students :}}\\ %Pour le titre au dessus des noms à gauche
        \eleves\\ %Remplacer pour chacun
		\end{flushleft}
	\end{minipage}
	~
	\begin{minipage}{0.4\textwidth}
		\begin{flushright} \large
		\emph{\textbf{Teacher :}} \\
		 \enseignant \\
		\end{flushright}
	\end{minipage}\\[4cm]
    
	\vspace{2.5cm}
    \begin{flushleft}
    \emph{\textbf{\textit{Code :}}} \url{https://github.com/thomas0barand/data-science-infologic.git}
    \end{flushleft}
    \vspace{1cm}
     
	{\large \today\par} %Affichage de la date

\end{titlepage}
}

\newcommand{\tabledematieres}{
\tableofcontents




\vfill
\noindent\textit{\small%
The project source code is available at: \url{https://github.com/thomas0barand/data-science-browser}.\\[4pt]
Submissions were made under the team name \textbf{"The best team"} on Kaggle: \url{https://www.kaggle.com/competitions/perfect-matches/overview}.\\
The contributions of each student to this project are as follows:
}


\begin{table}[b!]
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}ll@{}}
\specialrule{1.2pt}{0pt}{0pt}
\textbf{Élève} & \textbf{Contributions} \\
\specialrule{1pt}{0pt}{0pt}

Thomas Barand 
& Sparse embedding, TF-IDF, Bigrams, contribution contribution contribution \\ \midrule

Élève 2 
& contribution \\ \midrule

Élève 3 
& contribution \\

\specialrule{1.2pt}{0pt}{0pt}
\end{tabular}%
}
\end{table}



\newpage
}

\newcommand{\insererfigure}[4]{
\begin{figure}[ht]
\centering
\includegraphics[height=#2]{#1}
\caption{#3}
\label{fig: #4}
\end{figure}
}



