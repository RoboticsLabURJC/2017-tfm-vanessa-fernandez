SET( SOURCE_FILES objectdetector.cpp viewer.cpp)

add_definitions(-DGLADE_DIR="${gladedir}")

SET(PYTHON_LIBRARY "-L/usr/lib/python2.7/config-x86_64-linux-gnu/")
find_package( PythonInterp 2.7 REQUIRED )
find_package( PythonLibs 2.7 REQUIRED )

include_directories(
    ${INTERFACES_CPP_DIR}
    ${LIBS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${gtkmm_INCLUDE_DIRS}
    ${libglademm_INCLUDE_DIRS}
    ${gtkglextmm_INCLUDE_DIRS}
    ${ZLIB_INCLUDE_DIRS}
    ${resourcelocator_INCLUDE_DIRS}
    ${jderobottypes_INCLUDE_DIRS}
    ${comm_INCLUDE_DIRS}
    ${config_INCLUDE_DIRS}
    ${roscpp_INCLUDE_DIRS}
	${PYTHON_INCLUDE_DIRS}
)
link_directories(
	${resourcelocator_LIBRARY_DIRS}
    ${comm_LIBRARY_DIRS}
    ${config_LIBRARY_DIRS}
)

add_executable (objectdetector ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(objectdetector
    ${CMAKE_THREAD_LIBS_INIT} 
    ${OpenCV_LIBRARIES}
    ${OpenCVGUI_LIBRARIES}
    ${gtkmm_LIBRARIES}
    ${libglademm_LIBRARIES}
    colorspacesmm
    ${ZeroCIce_LIBRARIES}
    ${ZLIB_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
	${resourcelocator_LIBRARIES}
    ${catkin_LIBRARIES}
	${PYTHON_LIBRARIES}
    ${comm_LIBRARIES}
    ${config_LIBRARIES}
		${GLOG_LIBRARIES}
    JderobotInterfaces
        jderobotutil
		)

install(TARGETS objectdetector
    DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/
    COMPONENT objectdetector
    )


INSTALL (FILES ${CMAKE_CURRENT_SOURCE_DIR}/objectdetector.glade DESTINATION ${CMAKE_INSTALL_PREFIX}/share/jderobot/glade COMPONENT objectdetector)
INSTALL (FILES ${CMAKE_CURRENT_SOURCE_DIR}/objectdetector.yml DESTINATION ${CMAKE_INSTALL_PREFIX}/share/jderobot/conf COMPONENT objectdetector)
