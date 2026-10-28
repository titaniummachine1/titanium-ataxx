# Periodic chunk collector for Docker cloud sandboxes (tit-d1, tit-d2, tit-d3)
while ($true) {
    Start-Sleep -Seconds 180
    try {
        python "C:\gitProjects\titanium-ataxx\training\relabel_dis_cloud.py" collect --name tit-d1 --keep
    } catch {}
    try {
        python "C:\gitProjects\titanium-ataxx\training\relabel_dis_cloud.py" collect --name tit-d2 --keep
    } catch {}
    try {
        python "C:\gitProjects\titanium-ataxx\training\relabel_dis_cloud.py" collect --name tit-d3 --keep
    } catch {}
}
